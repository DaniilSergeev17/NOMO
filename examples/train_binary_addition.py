"""Train a character-level policy to emit a two-bit sum followed by EOS."""

from collections.abc import Sequence
from functools import partial
from itertools import product
from urllib.parse import quote

import torch
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from nomo import Decision, Episode, EpisodeEnd, Verification, VerificationStatus
from nomo.rloo import rloo_loss

OUTPUT_VOCAB = ('0', '1', '<eos>')
VOCAB = (*OUTPUT_VOCAB, '+', '=')
EOS = VOCAB.index('<eos>')
SAMPLING_REVISION = 'categorical-bits-eos/v1'
TASKS = tuple(product(range(2), repeat=2))
TextEpisode = Episode[tuple[int, ...], int]


class TrainConfig(BaseModel, frozen=True, extra='forbid'):
    """CPU experiment settings for the binary addition recipe."""

    seed: int = 3407
    eval_seed: int = 1337
    group_size: int = Field(default=128, ge=2)
    steps: int = Field(default=400, gt=0)
    learning_rate: float = Field(default=0.002, gt=0, allow_inf_nan=False)


class TinyLM(nn.Module):
    """Read prompts character by character and predict a bit or EOS."""

    def __init__(self) -> None:
        """Build an embedding, a single GRU layer, and a three-symbol output head."""
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 16)
        self.gru = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, len(OUTPUT_VOCAB))

    def forward(self, token_ids: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return next-token logits and the hidden state for incremental decoding."""
        embeddings = self.embedding(token_ids) # [batch_size, seq_len, embed_dim]
        # [batch_size, seq_len, hidden_size], [1, batch_size, hidden_size]:
        states, hidden = self.gru(embeddings, hidden)
        logits = self.head(states) # [batch_size, seq_len, output_vocab_size]
        return logits, hidden


@torch.no_grad()
def sample_group(
    model: TinyLM,
    left: int,
    right: int,
    episode_ids: tuple[str, ...],
    *,
    group_id: str,
    policy_version: str,
    generator: torch.Generator,
) -> list[TextEpisode]:
    """Sample one response per episode ID and score exact two-bit sums.

    EOS terminates a response. Hitting the three-token limit without EOS
    truncates the response and earns zero reward.
    """
    prompt = f'{left}+{right}='
    prompt_ids = tuple(VOCAB.index(char) for char in prompt)
    group_size = len(episode_ids)
    token_ids = torch.tensor(prompt_ids).expand(group_size, -1) # [group_size, prompt_len]
    hidden = None
    decisions: list[list[Decision[tuple[int, ...], int]]] = [[] for _ in episode_ids]
    prefixes = [prompt_ids] * group_size
    for _ in range(3): # two answer bits followed by EOS
        logits, hidden = model(token_ids, hidden)
        logprobs = logits[:, -1].log_softmax(-1) # [group_size, output_vocab_size]
        token_ids = torch.multinomial(logprobs.exp(), 1, generator=generator) # [group_size, 1]
        behavior_logprobs = logprobs.gather(1, token_ids).squeeze(1) # [group_size]
        sampled_actions = zip(token_ids.flatten().tolist(), behavior_logprobs.tolist(), strict=True)
        for i, (token, logprob) in enumerate(sampled_actions):
            if prefixes[i][-1] == EOS:
                continue
            decisions[i].append(Decision[tuple[int, ...], int](
                observation=prefixes[i], action=token, policy_version=policy_version,
                sampling_revision=SAMPLING_REVISION, behavior_logprob=logprob,
            ))
            prefixes[i] += (token,)
        if all(prefix[-1] == EOS for prefix in prefixes):
            break

    episodes = []
    for episode_id, steps in zip(episode_ids, decisions, strict=True):
        end = EpisodeEnd.TERMINATED if steps[-1].action == EOS else EpisodeEnd.TRUNCATED
        answer = ''.join(VOCAB[step.action] for step in steps if step.action != EOS)
        reward = float(end == EpisodeEnd.TERMINATED and answer == format(left + right, '02b'))
        episodes.append(TextEpisode(
            run_id='binary-addition', task_id=prompt, task_revision='addition/v1',
            environment_revision='bits-eos-3/v1', group_id=group_id,
            episode_id=episode_id, attempt_id='first', decisions=tuple(steps), end=end,
            verification=Verification(
                status=VerificationStatus.SCORED, reward=reward, verifier_revision='exact-binary/v1',
                evidence_ref='data:text/plain,' + quote(f'{prompt}{answer}\n{end.value}'),
            ),
        ))
    return episodes


def compute_current_logprobs(model: TinyLM, episodes: Sequence[TextEpisode]) -> Tensor:
    """Replay contiguous completions from sample_group in one causal forward.

    Returns:
        Differentiable logprobs in episode and decision order. Only positions
        that predict sampled actions contribute, including sampled EOS.
    """
    # each rollout in this recipe extends its prompt one token at a time
    inputs = [torch.tensor(ep.decisions[-1].observation) for ep in episodes]
    token_ids = pad_sequence(inputs, batch_first=True) # [group_size, max_seq_len]
    logits, _ = model(token_ids) # [group_size, max_seq_len, output_vocab_size]
    rows, positions, actions = [], [], []
    for row, ep in enumerate(episodes):
        for step in ep.decisions:
            rows.append(row)
            positions.append(len(step.observation) - 1)
            actions.append(step.action)
    action_logits = logits[rows, positions] # [action_count, output_vocab_size]
    return action_logits.log_softmax(-1).gather(1, torch.tensor(actions)[:, None]).squeeze(1) # [action_count]


def report_accuracy(model: TinyLM, episode_ids: tuple[str, ...], *, policy_version: str, seed: int) -> None:
    """Sample fresh completions on each training prompt with a separate generator."""
    generator = torch.Generator().manual_seed(seed)
    for left, right in TASKS:
        group = sample_group(
            model, left, right, episode_ids, group_id=f'eval/{policy_version}/{left}+{right}',
            policy_version=policy_version, generator=generator,
        )
        correct = sum(ep.verification.reward == 1.0 for ep in group)
        print(f'{policy_version}  {left}+{right}=  correct {correct}/{len(group)}')


def main() -> None:
    """Train on CPU and sample fresh groups from the final policy."""
    config = TrainConfig()
    print(f'torch={torch.__version__} config={config.model_dump_json()}')
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    model = TinyLM().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    episode_ids = tuple(str(i) for i in range(config.group_size))
    expected_ids = set(episode_ids)
    report_accuracy(model, episode_ids, policy_version='p0', seed=config.eval_seed)
    for step in range(config.steps):
        left, right = TASKS[step % len(TASKS)]
        version = f'p{step}'
        group = sample_group(
            model, left, right, episode_ids, group_id=f'train/{step}',
            policy_version=version, generator=generator,
        )
        loss = rloo_loss(
            group, partial(compute_current_logprobs, model),
            expected_episode_ids=expected_ids, current_policy_version=version,
            sampling_revision=SAMPLING_REVISION,
        )
        assert loss is not None # the synchronous verifier scored every episode
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    report_accuracy(model, episode_ids, policy_version=f'p{config.steps}', seed=config.eval_seed)


if __name__ == '__main__':
    main()

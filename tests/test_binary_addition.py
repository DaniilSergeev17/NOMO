from functools import partial

import torch

from examples.train_binary_addition import (
    EOS,
    SAMPLING_REVISION,
    VOCAB,
    TextEpisode,
    TinyLM,
    compute_current_logprobs,
    sample_group,
)
from nomo import EpisodeEnd
from nomo.rloo import rloo_loss


def test_generated_group_replays_and_updates_the_policy():
    seed = 3407
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TinyLM().eval()
    episode_ids = tuple(str(i) for i in range(128))
    generator = torch.Generator().manual_seed(seed)
    group = sample_group(model, 1, 1, episode_ids, group_id='first', policy_version='p0', generator=generator)
    group = [TextEpisode.model_validate_json(ep.model_dump_json()) for ep in group]

    assert {ep.end for ep in group} == {EpisodeEnd.TERMINATED, EpisodeEnd.TRUNCATED}
    assert {ep.verification.reward for ep in group} == {0.0, 1.0}
    for ep in group:
        prefix = tuple(VOCAB.index(char) for char in '1+1=')
        actions = tuple(step.action for step in ep.decisions)
        for step in ep.decisions:
            assert step.observation == prefix
            prefix += (step.action,)
        assert EOS not in actions[:-1]
        assert (ep.end == EpisodeEnd.TERMINATED) == (actions[-1] == EOS)
        if ep.end == EpisodeEnd.TRUNCATED:
            assert len(actions) == 3
        assert ep.verification.reward == float(actions == (1, 0, EOS))

    behavior_logprobs = torch.tensor([step.behavior_logprob for ep in group for step in ep.decisions]) # [action_count]
    torch.testing.assert_close(compute_current_logprobs(model, group), behavior_logprobs)
    loss = rloo_loss(
        group, partial(compute_current_logprobs, model), expected_episode_ids=set(episode_ids),
        current_policy_version='p0', sampling_revision=SAMPLING_REVISION,
    )
    assert loss is not None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss.backward()
    optimizer.step()
    assert not torch.allclose(compute_current_logprobs(model, group), behavior_logprobs)

    updated = sample_group(model, 1, 1, episode_ids, group_id='second', policy_version='p1', generator=generator)
    assert all(step.policy_version == 'p1' for ep in updated for step in ep.decisions)
    updated_logprobs = torch.tensor([step.behavior_logprob for ep in updated for step in ep.decisions]) # [action_count]
    torch.testing.assert_close(compute_current_logprobs(model, updated), updated_logprobs)

from math import log
from unittest.mock import Mock

import pytest
import torch

from nomo import Decision, Episode, EpisodeEnd, Verification, VerificationStatus
from nomo.rloo import rloo_loss


def episode(episode_id, actions, reward, *, group_id='g', policy_version='p1', sampling_revision='categorical'):
    return Episode[int, int](
        run_id='run', task_id='task', task_revision='v1', environment_revision='v1',
        group_id=group_id, episode_id=episode_id, attempt_id='first', end=EpisodeEnd.TERMINATED,
        decisions=tuple(
            Decision[int, int](
                observation=i, action=a, policy_version=policy_version, sampling_revision=sampling_revision,
            )
            for i, a in enumerate(actions)
        ),
        verification=Verification(
            verifier_revision='v1', evidence_ref=f'results/{episode_id}',
            status=VerificationStatus.SCORED if reward is not None else VerificationStatus.ERROR,
            reward=reward,
        ),
    )


def loss_for_pair(episodes, compute_current_logprobs):
    return rloo_loss(
        episodes, compute_current_logprobs, expected_episode_ids={'a', 'b'},
        current_policy_version='p1', sampling_revision='categorical',
    )


def test_cpu_update_matches_hand_calculation():
    group = [episode('a', [0, 0], 1.0), episode('b', [1], 0.0)]
    # [num_observations, num_action_choices]:
    logits = torch.nn.Parameter(torch.tensor([[log(3), 0.0], [0.0, 0.0]], dtype=torch.float64))
    optimizer = torch.optim.SGD([logits], lr=0.2)

    def compute_current_logprobs(episodes):
        decisions = [step for ep in episodes for step in ep.decisions]
        observations = [step.observation for step in decisions]
        actions = [step.action for step in decisions]
        # [num_observations, num_action_choices] -> [action_count]:
        return logits.log_softmax(dim=-1)[observations, actions]

    loss = loss_for_pair(group, compute_current_logprobs) # scalar
    # sequence probabilities are 3/4 * 1/2 and 1/4, with advantages +1 and -1
    assert loss.item() == pytest.approx(-log(1.5) / 2)
    loss.backward()
    torch.testing.assert_close(logits.grad, logits.new_tensor([[-0.5, 0.5], [-0.25, 0.25]]))
    optimizer.step()
    torch.testing.assert_close(logits, logits.new_tensor([[log(3) + 0.1, -0.1], [0.05, -0.05]]))


@pytest.mark.parametrize(
    'rewards, grad',
    [([0, 1, 4], [2.5 / 3, 1 / 3, -3.5 / 3]), ([0.1] * 3, [0.0] * 3)],
)
def test_leave_one_out_gradient(rewards, grad):
    group = [episode(str(i), [0], reward) for i, reward in enumerate(rewards)]
    current_logprobs = torch.tensor([-0.4, -0.8, -1.2], requires_grad=True) # [action_count]
    loss = rloo_loss(
        group, lambda _: current_logprobs, expected_episode_ids={'0', '1', '2'},
        current_policy_version='p1', sampling_revision='categorical',
    )
    loss.backward()
    torch.testing.assert_close(current_logprobs.grad, current_logprobs.new_tensor(grad))


def test_large_shared_reward_preserves_unit_difference():
    group = [episode('a', [0], 2**52), episode('b', [1], 2**52 + 1)]
    current_logprobs = torch.tensor([-0.3, -1.3], dtype=torch.float32, requires_grad=True) # [action_count]
    loss_for_pair(group, lambda _: current_logprobs).backward()
    torch.testing.assert_close(current_logprobs.grad, torch.tensor([0.5, -0.5]))


@pytest.mark.parametrize('pending', [[], [episode('b', [1], None)]])
def test_group_waits_for_all_verified_rewards(pending):
    compute_current_logprobs = Mock()
    assert loss_for_pair([episode('a', [0], 1.0), *pending], compute_current_logprobs) is None
    compute_current_logprobs.assert_not_called()


@pytest.mark.parametrize(
    'other, message',
    [
        (episode('a', [1], 0.0), 'duplicate'),
        (episode('unexpected', [1], 0.0), 'unexpected'),
        (episode('b', [1], 0.0, group_id='other'), 'same run'),
        (episode('b', [1], 0.0, policy_version='p0'), 'weights'),
        (episode('b', [1], 0.0, sampling_revision='greedy'), 'sampling'),
    ],
)
def test_invalid_group_is_rejected_before_scoring(other, message):
    compute_current_logprobs = Mock()
    with pytest.raises(ValueError, match=message):
        loss_for_pair([episode('a', [0], 1.0), other], compute_current_logprobs)
    compute_current_logprobs.assert_not_called()


@pytest.mark.parametrize(
    'current_logprobs, message',
    [(torch.zeros(2), 'per decision'), (torch.tensor([0., -torch.inf, 0.]), 'non-finite')],
)
def test_invalid_current_logprobs(current_logprobs, message):
    group = [episode('a', [0, 0], 1.0), episode('b', [1], 0.0)]
    with pytest.raises(ValueError, match=message):
        loss_for_pair(group, lambda _: current_logprobs)


def test_group_without_policy_actions_skips_scoring():
    compute_current_logprobs = Mock()
    assert loss_for_pair([episode('a', [], 1.0), episode('b', [], 0.0)], compute_current_logprobs) is None
    compute_current_logprobs.assert_not_called()

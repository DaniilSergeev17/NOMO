"""On-policy REINFORCE leave-one-out (RLOO) loss for verified rollouts."""

from collections.abc import Callable, Sequence, Set
from statistics import fmean

import torch
from torch import Tensor

from .episode import Action, Episode, Observation


def rloo_loss(
    episodes: Sequence[Episode[Observation, Action]],
    compute_current_logprobs: Callable[[Sequence[Episode[Observation, Action]]], Tensor],
    *,
    expected_episode_ids: Set[str],
    current_policy_version: str,
    sampling_revision: str,
) -> Tensor | None:
    """Compute the on-policy RLOO loss for one group of independent rollouts.

    Each episode's advantage is its reward minus the mean reward of its peers.
    Use the group for one update with the policy that generated it.

    Args:
        episodes: Available attempts for the same run, task, and rollout group.
        compute_current_logprobs: Function that evaluates the recorded actions
            on their original observations using the specified weights and
            sampling rule. It returns a differentiable tensor of shape
            (action_count,), with one joint log probability or log density per
            decision, in episode order and then decision order.
        expected_episode_ids: Episode IDs assigned to the group before sampling.
        current_policy_version: Version of the weights used to compute logprobs.
        sampling_revision: Identifier for the sampling rule and its parameters.

    Returns:
        Scalar loss summed over actions within each episode and averaged over
        episodes. Returns None if any expected episode or reward is missing,
        or if the group contains no actions.

    Raises:
        ValueError: If fewer than two episodes are expected, IDs are duplicated
            or unexpected, or the episodes come from different groups or
            revisions. Also raised if a decision uses different weights or
            sampling rules, or the logprob tensor has the wrong shape,
            a non-floating dtype, or non-finite values.

    References:
        https://huggingface.co/papers/2402.14740
    """
    group_size = len(expected_episode_ids)
    if group_size < 2:
        raise ValueError('RLOO requires at least two expected episodes')
    episode_ids = {ep.episode_id for ep in episodes}
    if len(episode_ids) != len(episodes) or not episode_ids.issubset(expected_episode_ids):
        raise ValueError('Received a duplicate or unexpected episode ID')
    group_keys = {
        (
            ep.run_id, ep.task_id, ep.task_revision, ep.group_id,
            ep.environment_revision, ep.verification.verifier_revision,
        )
        for ep in episodes
    }
    if len(group_keys) > 1:
        raise ValueError('Episodes must belong to the same run, task, group and revisions')
    rewards = [ep.verification.reward for ep in episodes if ep.verification.reward is not None]
    if len(rewards) != group_size:
        return None
    for ep in episodes:
        for decision in ep.decisions:
            if (
                decision.policy_version != current_policy_version
                or decision.sampling_revision != sampling_revision
            ):
                raise ValueError('A decision was sampled with different weights or sampling rules')

    episode_lengths = [len(ep.decisions) for ep in episodes]
    action_count = sum(episode_lengths)
    if action_count == 0:
        return None
    current_logprobs = compute_current_logprobs(episodes)  # [action_count]
    if current_logprobs.shape != (action_count,) or not current_logprobs.is_floating_point():
        raise ValueError('compute_current_logprobs must return one floating-point logprob per decision')
    current_logprobs = current_logprobs.to(torch.promote_types(current_logprobs.dtype, torch.float32))
    if not torch.isfinite(current_logprobs).all():  # [action_count] -> scalar
        raise ValueError('An observed action has a non-finite logprob under the current policy')

    # R_i - mean(R_others) = (R_i - mean(R)) * G / (G - 1).
    # Subtracting the first reward keeps a large shared value out of the mean.
    shifted_rewards = [reward - rewards[0] for reward in rewards]
    mean_shifted_reward = fmean(shifted_rewards)
    episode_advantages = current_logprobs.new_tensor([
        (reward - mean_shifted_reward) * (group_size / (group_size - 1)) for reward in shifted_rewards
    ])  # [group_size]
    action_advantages = episode_advantages.repeat_interleave(
        torch.tensor(episode_lengths, device=current_logprobs.device), output_size=action_count,
    )  # [group_size] -> [action_count]
    # [action_count] @ [action_count] -> scalar:
    return -torch.dot(current_logprobs, action_advantages) / group_size

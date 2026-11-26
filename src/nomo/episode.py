"""Records shared by rollout, verification, and training."""

from enum import StrEnum
from typing import Annotated, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


Observation = TypeVar('Observation')
Action = TypeVar('Action')
Identifier = Annotated[str, Field(pattern=r'\S')]


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True, allow_inf_nan=False)


class EpisodeEnd(StrEnum):
    """Distinguish a terminal environment state from an external cutoff."""

    TERMINATED = 'terminated'
    TRUNCATED = 'truncated'


class VerificationStatus(StrEnum):
    """Indicate whether the verifier produced a usable reward."""

    SCORED = 'scored'
    ERROR = 'error'


class Verification(_Record):
    """Verifier outcome for a whole episode.

    Attributes:
        verifier_revision: Version of the verifier used for this episode.
        evidence_ref: Reference to the evidence for this result.
        status: Whether the verifier scored the episode or encountered an error.
        reward: Finite score for the episode, or None on verifier error.
    """

    verifier_revision: Identifier
    evidence_ref: Identifier
    status: VerificationStatus
    reward: float | None

    @model_validator(mode='after')
    def check_reward(self) -> Self:
        """Require a reward for a scored episode and None for a verifier error."""
        if (self.status == VerificationStatus.SCORED) != (self.reward is not None):
            raise ValueError('Scored attempts require a reward and verifier errors require null')
        return self


class Decision(_Record, Generic[Observation, Action]):
    """One policy step: an observation and the action sampled from it.

    In autoregressive text generation, the observation is the context seen
    so far and the action is the next generated token.

    Attributes:
        observation: Exact input the policy saw when choosing the action.
        action: Action the policy took for that observation.
        policy_version: Version of the weights used to sample the action.
        sampling_revision: Identifier for the sampling rule and its parameters.
        behavior_logprob: Joint log probability (or log density) of the action
            after sampling transforms, or None if the sampler did not record it.
    """

    observation: Observation
    action: Action
    policy_version: Identifier
    sampling_revision: Identifier
    behavior_logprob: float | None = None


class Episode(_Record, Generic[Observation, Action]):
    """One rollout from the first observation to termination or truncation.

    In single-turn text generation, one episode records one response to a
    prompt, up to the stop condition or token limit.

    Use the adapter's observation and action types when loading an episode from JSON.

    Attributes:
        run_id: Training run that requested the episode.
        task_id: Task assigned to the policy.
        task_revision: Version of the task specification.
        environment_revision: Version of the environment used for this rollout.
        group_id: Group of rollouts sampled for the same task.
        episode_id: Identifier of this episode within the group, retained on retry.
        attempt_id: Identifier for one execution of this episode. A retry gets
            a new attempt_id and keeps the same episode_id.
        decisions: Policy decisions in the order they occurred.
        end: Whether the episode terminated or was truncated.
        verification: Verifier outcome and associated evidence.
    """

    run_id: Identifier
    task_id: Identifier
    task_revision: Identifier
    environment_revision: Identifier
    group_id: Identifier
    episode_id: Identifier
    attempt_id: Identifier
    decisions: tuple[Decision[Observation, Action], ...]
    end: EpisodeEnd
    verification: Verification

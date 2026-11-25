from enum import StrEnum
from typing import Annotated, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


Observation = TypeVar('Observation')
Action = TypeVar('Action')
Identifier = Annotated[str, Field(pattern=r'\S')]


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', frozen=True, allow_inf_nan=False)


class EpisodeEnd(StrEnum):
    TERMINATED = 'terminated'
    TRUNCATED = 'truncated'


class VerificationStatus(StrEnum):
    SCORED = 'scored'
    ERROR = 'error'


class Verification(_Record):
    verifier_revision: Identifier
    evidence_ref: Identifier
    status: VerificationStatus
    reward: float | None

    @model_validator(mode='after')
    def check_reward(self) -> Self:
        if (self.status == VerificationStatus.SCORED) != (self.reward is not None):
            raise ValueError('Scored attempts require a reward and verifier errors require null')
        return self


class Decision(_Record, Generic[Observation, Action]):
    '''One action and the exact input seen by the behavior policy.

    sampling_revision identifies the sampling rule and its parameters.
    behavior_logprob, when present, is the joint log probability (or log
    density) of the recorded action under that rule after sampling transforms.
    '''

    observation: Observation
    action: Action
    policy_version: Identifier
    sampling_revision: Identifier
    behavior_logprob: float | None = None


class Episode(_Record, Generic[Observation, Action]):
    '''A completed attempt. Use the adapter's observation and action types when loading JSON.'''

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

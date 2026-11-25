import json

import pytest
from pydantic import NonNegativeInt, ValidationError

from nomo import Episode, EpisodeEnd, Verification, VerificationStatus


TokenEpisode = Episode[tuple[NonNegativeInt, ...], NonNegativeInt]
Vector = tuple[float, float]
ControlEpisode = Episode[Vector, Vector]


@pytest.fixture
def attempt():
    return {
        'run_id': 'run-1',
        'group_id': 'group-1',
        'episode_id': 'episode-1',
        'attempt_id': 'attempt-1',
    }


@pytest.fixture
def token_record(attempt):
    sampling = {'sampling_revision': 'greedy/v1', 'behavior_logprob': 0.0}
    return {
        **attempt,
        'task_id': 'copy',
        'task_revision': 'copy/v1',
        'environment_revision': 'tokens/v1',
        'decisions': [
            {**sampling, 'observation': [7, 4], 'action': 7, 'policy_version': 'p1'},
            {**sampling, 'observation': [7, 4, 7], 'action': 4, 'policy_version': 'p2'},
        ],
        'end': 'terminated',
        'verification': {
            'verifier_revision': 'exact-copy/v1',
            'evidence_ref': 'results/copy/1',
            'status': 'scored',
            'reward': 1.0,
        },
    }


def test_replay_token_episode(token_record):
    episode = TokenEpisode.model_validate_json(json.dumps(token_record))
    restored = TokenEpisode.model_validate_json(episode.model_dump_json())
    assert restored == episode

    prompt = (7, 4)
    generated = []
    for step in restored.decisions:
        assert step.observation == prompt + tuple(generated)
        generated.append(step.action)
    assert restored.verification.reward == float(tuple(generated) == prompt) == 1.0
    assert [step.policy_version for step in restored.decisions] == ['p1', 'p2']
    assert restored.end == EpisodeEnd.TERMINATED


def test_replay_continuous_episode(attempt):
    sampling = {'policy_version': 'p1', 'sampling_revision': 'gaussian/v1'}
    record = {
        **attempt,
        'task_id': 'reach-origin',
        'task_revision': 'reach-origin/v1',
        'environment_revision': 'point/v1',
        'decisions': [
            {**sampling, 'observation': [1.0, -0.5], 'action': [-0.25, 0.125], 'behavior_logprob': 1.25},
            {**sampling, 'observation': [0.75, -0.375], 'action': [-0.25, 0.25], 'behavior_logprob': 0.5},
        ],
        'end': 'truncated',
        'verification': {
            'verifier_revision': 'at-origin/v1',
            'evidence_ref': 'results/point/1',
            'status': 'scored',
            'reward': 0.0,
        },
    }
    episode = ControlEpisode.model_validate_json(json.dumps(record))
    restored = ControlEpisode.model_validate_json(episode.model_dump_json())
    assert restored == episode

    x, y = 1.0, -0.5
    for step in restored.decisions:
        assert step.observation == (x, y)
        dx, dy = step.action
        x, y = x + dx, y + dy
    assert (x, y) == (0.5, -0.125)
    assert restored.verification.reward == float((x, y) == (0.0, 0.0)) == 0.0
    assert restored.decisions[0].behavior_logprob == 1.25
    assert restored.end == EpisodeEnd.TRUNCATED


def test_verifier_error_can_be_recorded_without_reward(token_record):
    token_record['verification'].update(status='error', reward=None)
    result = TokenEpisode.model_validate_json(json.dumps(token_record)).verification
    assert result.status == VerificationStatus.ERROR
    assert result.reward is None


@pytest.mark.parametrize('status, reward', [('error', 0.0), ('scored', None), ('scored', float('inf'))])
def test_invalid_verification(status, reward):
    with pytest.raises(ValidationError):
        Verification(
            verifier_revision='v1', evidence_ref='result/1',
            status=VerificationStatus(status), reward=reward,
        )


@pytest.mark.parametrize('action', [-1, 3.5, '7', True, [7]])
def test_invalid_token_action(token_record, action):
    token_record['decisions'][0]['action'] = action
    with pytest.raises(ValidationError, match='action'):
        TokenEpisode.model_validate_json(json.dumps(token_record))

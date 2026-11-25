"""NOMO, a framework for reinforcement learning with verifiable rewards."""

from importlib.metadata import version as _version

from .episode import Decision, Episode, EpisodeEnd, Verification, VerificationStatus

__version__ = _version('nomo-rl')
__all__ = ['Decision', 'Episode', 'EpisodeEnd', 'Verification', 'VerificationStatus', '__version__']

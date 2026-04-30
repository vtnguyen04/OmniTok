"""Alignment loss strategies for VFM feature alignment."""

from .base import BaseAlignmentLoss

# Import to trigger ALIGNMENT_REGISTRY registration
from . import cosine  # noqa: F401
from . import relational  # noqa: F401
from . import prediction  # noqa: F401

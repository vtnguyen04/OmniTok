"""Alignment loss strategies for VFM feature alignment."""

# Import to trigger ALIGNMENT_REGISTRY registration
from . import (
    cosine,  # noqa: F401
    prediction,  # noqa: F401
    relational,  # noqa: F401
)
from .base import BaseAlignmentLoss

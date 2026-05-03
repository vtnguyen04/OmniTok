"""Loss modules for OmniTok training."""

# Import alignment subpackage to trigger registrations
from . import alignment  # noqa: F401
from .gan import GANLoss
from .gaussianity import GaussianityLoss
from .kl import KLLoss
from .latent_norm import LatentNormLoss
from .reconstruction import ReconstructionLoss

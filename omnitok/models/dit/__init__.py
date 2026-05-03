from omnitok.registry import Registry

from .lightning_dit import LightningDiT_models
from .sit import SiT_models

DIT_REGISTRY = Registry("DiT")

for name, model_fn in SiT_models.items():
    DIT_REGISTRY.register(name)(model_fn)

for name, model_fn in LightningDiT_models.items():
    DIT_REGISTRY.register(name)(model_fn)

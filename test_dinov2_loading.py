import torch
import logging
logging.basicConfig(level=logging.INFO)
from omnitok.registry import ENCODER_REGISTRY
import omnitok.models.encoder.vision_transformer_bottleneck

# Enable logger for omnitok
logging.getLogger('omnitok.models.encoder.vision_transformer_bottleneck').setLevel(logging.INFO)

encoder = ENCODER_REGISTRY.build(
    'dinov2_bottleneck',
    img_size=256,
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    vit_feature_bottleneck=32
)

encoder.load_pretrained_dinov2('dinov2_vitb14')

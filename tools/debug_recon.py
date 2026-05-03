"""Debug script: diagnose why reconstruction outputs are constant/structured noise.

Checks:
1. Are encoder outputs different for different inputs?
2. Are latents (after bottleneck) different for different inputs?
3. Are decoder outputs different for different inputs?
4. Gradient flow through encoder → bottleneck → decoder
5. Statistics at each stage
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path


def main():
    from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
    from omnitok.models.decoder.pixel_decoder import DinoV3PixelDecoder
    from omnitok.models.tokenizer import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model (same as T2 config)
    encoder = DinoVisionTransformerWithBottleneck(
        img_size=256, patch_size=16, embed_dim=768,
        depth=12, num_heads=12, vit_feature_bottleneck=32,
    )
    # Load pretrained
    encoder.load_pretrained_dinov2("dinov2_vitb14")

    decoder = DinoV3PixelDecoder(
        in_chans=32, out_chans=3, upscale_factor=16,
        embed_dim=1024, depth=12, num_heads=16,
    )

    tokenizer = Tokenizer(encoder=encoder, decoder=decoder).to(device)

    # Create 4 very different synthetic inputs
    torch.manual_seed(42)
    img_black = torch.zeros(1, 3, 256, 256).to(device)      # all black
    img_white = torch.ones(1, 3, 256, 256).to(device)        # all white
    img_rand1 = torch.randn(1, 3, 256, 256).to(device)      # random 1
    img_rand2 = torch.randn(1, 3, 256, 256).to(device) * 2  # random 2 (different)

    # Also load a real image if available
    data_dir = Path(os.path.expanduser("~/Tan/AIOTok/data/mini"))
    real_imgs = []
    if data_dir.exists():
        for img_path in sorted(data_dir.rglob("*.JPEG"))[:2]:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            real_imgs.append(img)

    inputs = {
        "black": img_black,
        "white": img_white,
        "rand1": img_rand1,
        "rand2": img_rand2,
    }
    for i, img in enumerate(real_imgs):
        inputs[f"real_{i}"] = img

    print("\n" + "="*80)
    print("STAGE 1: Encoder output (BEFORE bottleneck)")
    print("="*80)
    with torch.no_grad():
        for name, img in inputs.items():
            feat = encoder.forward_features(img, use_bottleneck=False)
            patch = feat["x_norm_patchtokens"]
            cls = feat["x_norm_clstoken"]
            print(f"  {name:8s} | patch shape={patch.shape} mean={patch.mean():.4f} std={patch.std():.4f} "
                  f"min={patch.min():.4f} max={patch.max():.4f} | cls norm={cls.norm():.4f}")

    print("\n" + "="*80)
    print("STAGE 2: Latent (AFTER bottleneck, before decoder)")
    print("="*80)
    latents = {}
    with torch.no_grad():
        for name, img in inputs.items():
            z = tokenizer.encode(img)
            latents[name] = z
            print(f"  {name:8s} | shape={z.shape} mean={z.mean():.6f} std={z.std():.6f} "
                  f"min={z.min():.4f} max={z.max():.4f}")

    # Check if latents are DIFFERENT from each other
    print("\n  Pairwise L2 distances between latents:")
    names = list(latents.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            dist = (latents[names[i]] - latents[names[j]]).norm().item()
            print(f"    {names[i]:8s} vs {names[j]:8s}: L2={dist:.6f}")

    print("\n" + "="*80)
    print("STAGE 3: Decoder output")
    print("="*80)
    recons = {}
    with torch.no_grad():
        for name, z in latents.items():
            recon = tokenizer.decode(z)
            recons[name] = recon
            print(f"  {name:8s} | shape={recon.shape} mean={recon.mean():.6f} std={recon.std():.6f} "
                  f"min={recon.min():.4f} max={recon.max():.4f}")

    # Check if decoder outputs are DIFFERENT from each other
    print("\n  Pairwise L2 distances between reconstructions:")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            dist = (recons[names[i]] - recons[names[j]]).norm().item()
            print(f"    {names[i]:8s} vs {names[j]:8s}: L2={dist:.6f}")

    print("\n" + "="*80)
    print("STAGE 4: Gradient flow check")
    print("="*80)
    tokenizer.train()
    img = inputs["rand1"].requires_grad_(False)
    output = tokenizer(img)
    recon = output["reconstruction"]
    loss = nn.functional.l1_loss(recon, img)
    loss.backward()

    # Check gradients
    layers_to_check = [
        ("encoder.feature_bottleneck", encoder.feature_bottleneck),
        ("decoder.proj_in", decoder.proj_in),
        ("decoder.blocks[0].attn.qkv", decoder.blocks[0].attn.qkv),
        ("decoder.blocks[-1].attn.qkv", decoder.blocks[-1].attn.qkv),
        ("decoder.proj_out", decoder.proj_out),
        ("decoder.norm", decoder.norm),
    ]
    for name, module in layers_to_check:
        if module is None:
            print(f"  {name:40s} | module is None!")
            continue
        if hasattr(module, 'weight') and module.weight.grad is not None:
            g = module.weight.grad
            print(f"  {name:40s} | grad norm={g.norm():.8f} mean={g.mean():.8f} std={g.std():.8f}")
        else:
            print(f"  {name:40s} | NO GRADIENT!")

    # Check encoder block gradients
    for i, blk in enumerate(encoder.blocks):
        if blk.attn.qkv.weight.grad is not None:
            g = blk.attn.qkv.weight.grad
            if i == 0 or i == len(encoder.blocks)-1:
                print(f"  encoder.blocks[{i}].attn.qkv        | grad norm={g.norm():.8f}")
        else:
            if i == 0 or i == len(encoder.blocks)-1:
                print(f"  encoder.blocks[{i}].attn.qkv        | NO GRADIENT!")

    print("\n" + "="*80)
    print("STAGE 5: Decoder internal check - does proj_in produce different outputs?")
    print("="*80)
    with torch.no_grad():
        for name, z in latents.items():
            proj = decoder.proj_in(z)
            print(f"  {name:8s} after proj_in | mean={proj.mean():.6f} std={proj.std():.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

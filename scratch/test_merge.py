from omegaconf import OmegaConf

base = OmegaConf.create({
    "model": {
        "bottleneck": {"latent_dim": 32, "type": "linear"},
        "decoder": {"in_chans": 256}
    },
    "experiment": {
        "model": {
            "bottleneck": {"latent_dim": 64, "type": "variational"},
            "decoder": {"in_chans": 64}
        }
    }
})

merged = OmegaConf.merge(base, base.experiment)
print("Merged latent_dim:", merged.model.bottleneck.latent_dim)
print("Merged in_chans:", merged.model.decoder.in_chans)
print("Merged type:", merged.model.bottleneck.type)

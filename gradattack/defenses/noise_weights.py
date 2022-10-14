import torch


def apply_noise(model, clipping_bound, noise_multiplier, deviceG):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()).to(torch.device(f"cuda:{deviceG}")) * (noise_multiplier **0.5))

    return model
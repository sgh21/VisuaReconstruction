from __future__ import annotations

import torch
from torchvision.transforms import functional as TF


def _random_erase(x: torch.Tensor, max_fraction: float = 0.18, prob: float = 0.7) -> torch.Tensor:
    out = x.clone()
    batch, _, height, width = out.shape
    for i in range(batch):
        if torch.rand((), device=out.device).item() > prob:
            continue
        erase_h = int(height * torch.empty((), device=out.device).uniform_(0.05, max_fraction).item())
        erase_w = int(width * torch.empty((), device=out.device).uniform_(0.05, max_fraction).item())
        if erase_h < 1 or erase_w < 1:
            continue
        top = int(torch.randint(0, max(1, height - erase_h + 1), (), device=out.device).item())
        left = int(torch.randint(0, max(1, width - erase_w + 1), (), device=out.device).item())
        value = torch.rand((3, 1, 1), device=out.device) * 0.15
        out[i, :, top : top + erase_h, left : left + erase_w] = value
    return out


def _patch_mask(x: torch.Tensor, patch_size: int = 16, mask_ratio: float = 0.35) -> torch.Tensor:
    out = x.clone()
    batch, channels, height, width = out.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    if grid_h == 0 or grid_w == 0:
        return out
    for i in range(batch):
        mask = torch.rand((grid_h, grid_w), device=out.device) < mask_ratio
        expanded = mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
        expanded = expanded[:height, :width]
        fill = out[i].mean(dim=(1, 2), keepdim=True)
        out[i] = torch.where(expanded.unsqueeze(0), fill.expand(channels, height, width), out[i])
    return out


def degrade_batch(x: torch.Tensor, mode: str = "blur_noise_erase") -> torch.Tensor:
    """Create degraded clean inputs for clean-prior self-supervision."""
    out = x.clone()

    if "blur" in mode:
        kernel = 5 if min(out.shape[-2:]) < 384 else 7
        sigma = float(torch.empty((), device=out.device).uniform_(0.5, 1.8).item())
        out = TF.gaussian_blur(out, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

    if "noise" in mode:
        noise_std = float(torch.empty((), device=out.device).uniform_(0.01, 0.045).item())
        out = out + torch.randn_like(out) * noise_std

    scale = torch.empty((out.shape[0], 1, 1, 1), device=out.device).uniform_(0.85, 1.15)
    bias = torch.empty((out.shape[0], 1, 1, 1), device=out.device).uniform_(-0.04, 0.04)
    out = out * scale + bias

    if mode == "blur_noise_erase":
        out = _random_erase(out)
    elif mode == "blur_noise_patchmask":
        out = _patch_mask(out)
    elif mode == "mild":
        pass
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    return out.clamp(0.0, 1.0)


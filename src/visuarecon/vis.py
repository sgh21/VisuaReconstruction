from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    array = mask.detach().cpu().clamp(0, 1).squeeze().numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def valid_fov_mask(image: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    gray = image.mean(dim=0, keepdim=True)
    return (gray > threshold).float()


def residual_mask(
    image: torch.Tensor,
    prior_hat: torch.Tensor,
    p_low: float = 5.0,
    p_high: float = 95.0,
) -> torch.Tensor:
    diff = (image - prior_hat).abs().mean(dim=0, keepdim=True)
    fov = valid_fov_mask(image)
    values = diff[fov.bool()]
    if values.numel() < 32:
        values = diff.flatten()
    low = torch.quantile(values, p_low / 100.0)
    high = torch.quantile(values, p_high / 100.0)
    mask = (diff - low) / (high - low + 1e-6)
    return mask.clamp(0, 1) * fov


def heatmap_rgb(mask: torch.Tensor, cmap_name: str = "inferno") -> np.ndarray:
    mask_np = mask.detach().cpu().clamp(0, 1).squeeze().numpy()
    mapper = cm.get_cmap(cmap_name)
    heat = mapper(mask_np)[..., :3]
    return (heat * 255.0 + 0.5).astype(np.uint8)


def overlay_mask(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.35) -> np.ndarray:
    base = tensor_to_uint8_image(image).astype(np.float32)
    heat = heatmap_rgb(mask).astype(np.float32)
    out = base * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_suction_visuals(
    image: torch.Tensor,
    prior_hat: torch.Tensor,
    mask: torch.Tensor,
    output_stem: Path,
    alpha: float = 0.35,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    suction_np = tensor_to_uint8_image(image)
    prior_np = tensor_to_uint8_image(prior_hat)
    mask_np = mask_to_uint8(mask)
    heat_np = heatmap_rgb(mask)
    overlay_np = overlay_mask(image, mask, alpha=alpha)

    Image.fromarray(suction_np).save(output_stem.with_name(output_stem.name + "_suction.png"))
    Image.fromarray(prior_np).save(output_stem.with_name(output_stem.name + "_prior.png"))
    Image.fromarray(mask_np).save(output_stem.with_name(output_stem.name + "_mask.png"))
    Image.fromarray(overlay_np).save(output_stem.with_name(output_stem.name + "_overlay.png"))

    separator = np.full((suction_np.shape[0], 8, 3), 255, dtype=np.uint8)
    mask_rgb = np.repeat(mask_np[..., None], 3, axis=2)
    grid = np.concatenate([suction_np, separator, prior_np, separator, heat_np, separator, overlay_np, separator, mask_rgb], axis=1)
    Image.fromarray(grid).save(output_stem.with_name(output_stem.name + "_grid.png"))


def append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


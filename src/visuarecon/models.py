from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
)
from torchvision.models import ViT_B_16_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: str
    image_size: tuple[int, int]


class SegmentationRestorationWrapper(nn.Module):
    def __init__(self, model: nn.Module, normalize: bool) -> None:
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model_input = (x - self.mean) / self.std if self.normalize else x
        out = self.model(model_input)["out"]
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


class MaeVitRestoration(nn.Module):
    """MAE-style clean-prior using torchvision ViT as the encoder backbone."""

    def __init__(
        self,
        weights: str = "default",
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.35,
    ) -> None:
        super().__init__()
        if weights == "default":
            if image_size != 224:
                raise ValueError("mae_vit_b_16 with default torchvision weights requires 224x224 inputs.")
            vit_weights = ViT_B_16_Weights.DEFAULT
            normalize = True
        elif weights == "none":
            vit_weights = None
            normalize = False
        else:
            raise ValueError(f"Unknown weights option: {weights}")

        self.vit = models.vit_b_16(weights=vit_weights, image_size=image_size)
        self.normalize = normalize
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        hidden_dim = self.vit.hidden_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_size * patch_size * 3),
        )
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        # Classification heads are not used for pixel reconstruction.
        self.vit.heads = nn.Identity()

    def _mask_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_ratio <= 0:
            return tokens
        batch, seq_len, hidden_dim = tokens.shape
        mask = torch.rand((batch, seq_len, 1), device=tokens.device) < self.mask_ratio
        return torch.where(mask, self.mask_token.expand(batch, seq_len, hidden_dim), tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"mae_vit_b_16 requires square {self.image_size}x{self.image_size} inputs, "
                f"got {w}x{h}."
            )
        model_input = (x - self.mean) / self.std if self.normalize else x
        tokens = self.vit._process_input(model_input)
        tokens = self._mask_tokens(tokens)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        tokens = torch.cat([batch_class_token, tokens], dim=1)
        tokens = self.vit.encoder(tokens)
        patch_tokens = tokens[:, 1:, :]
        patches = self.patch_decoder(patch_tokens)

        grid = self.image_size // self.patch_size
        patches = patches.view(n, grid, grid, 3, self.patch_size, self.patch_size)
        out = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(n, 3, self.image_size, self.image_size)
        return torch.sigmoid(out)


def _replace_last_conv(module: nn.Sequential, out_channels: int = 3) -> None:
    last = module[-1]
    if not isinstance(last, nn.Conv2d):
        raise TypeError(f"Expected final Conv2d, got {type(last)!r}")
    module[-1] = nn.Conv2d(last.in_channels, out_channels, kernel_size=last.kernel_size)


def build_model(name: str, weights: str, image_size: tuple[int, int]) -> nn.Module:
    if weights not in {"default", "none"}:
        raise ValueError("--weights must be 'default' or 'none'")

    if name == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(
            weights=FCN_ResNet50_Weights.DEFAULT if weights == "default" else None,
            weights_backbone=None,
            aux_loss=True if weights == "default" else False,
        )
        _replace_last_conv(model.classifier, 3)
        model.aux_classifier = None
        return SegmentationRestorationWrapper(model, normalize=weights == "default")

    if name == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT if weights == "default" else None,
            weights_backbone=None,
            aux_loss=True if weights == "default" else False,
        )
        _replace_last_conv(model.classifier, 3)
        model.aux_classifier = None
        return SegmentationRestorationWrapper(model, normalize=weights == "default")

    if name == "lraspp_mobilenet_v3_large":
        model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if weights == "default" else None,
            weights_backbone=None,
        )
        model.classifier.low_classifier = nn.Conv2d(
            model.classifier.low_classifier.in_channels, 3, kernel_size=1
        )
        model.classifier.high_classifier = nn.Conv2d(
            model.classifier.high_classifier.in_channels, 3, kernel_size=1
        )
        return SegmentationRestorationWrapper(model, normalize=weights == "default")

    if name == "mae_vit_b_16":
        width, height = image_size
        if width != height:
            raise ValueError("mae_vit_b_16 requires square --image-size, for example 224 224.")
        if width % 16 != 0:
            raise ValueError("mae_vit_b_16 image size must be divisible by 16.")
        return MaeVitRestoration(weights=weights, image_size=width)

    raise ValueError(f"Unknown model: {name}")


def official_image_size(name: str) -> tuple[int, int]:
    if name in {"fcn_resnet50", "deeplabv3_resnet50", "lraspp_mobilenet_v3_large"}:
        # DataLoader first center-crops 1920x1080 images to 1080x1080.
        # Torchvision segmentation weights use resize_size=[520], so the
        # square crop should be resized to 520x520 for official alignment.
        return (520, 520)
    if name == "mae_vit_b_16":
        return (224, 224)
    raise ValueError(f"Unknown model: {name}")


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    spec: ModelSpec,
) -> dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "model_name": spec.name,
        "weights": spec.weights,
        "image_size": list(spec.image_size),
    }

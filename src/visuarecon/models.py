from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed
from torchvision import models
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAE_VISUALIZE_VIT_BASE_URL = "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth"


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


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Sin-cos embedding dimension must be even.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("2D sin-cos embedding dimension must be even.")
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


class OfficialMaeVitRestoration(nn.Module):
    """MAE ViT-B/16 architecture aligned with the official asymmetric encoder-decoder."""

    def __init__(
        self,
        weights: str = "default",
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        if weights not in {"default", "none"}:
            raise ValueError(f"Unknown weights option: {weights}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
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

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

        self.initialize_weights()
        if weights == "default":
            self.load_official_visualize_weights()

    def initialize_weights(self) -> None:
        grid_size = int(self.patch_embed.num_patches**0.5)
        pos_embed = _get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        decoder_pos_embed = _get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def load_official_visualize_weights(self) -> None:
        print(f"Loading official MAE visualization checkpoint: {MAE_VISUALIZE_VIT_BASE_URL}")
        checkpoint = torch.hub.load_state_dict_from_url(
            MAE_VISUALIZE_VIT_BASE_URL,
            map_location="cpu",
            check_hash=False,
        )
        if isinstance(checkpoint, dict):
            print(f"Official MAE checkpoint top-level keys: {sorted(checkpoint.keys())}")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        msg = self.load_state_dict(state_dict, strict=True)
        print(f"Official MAE load_state_dict msg: {msg}")

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = imgs.shape[2] // p
        if imgs.shape[2] != imgs.shape[3] or imgs.shape[2] % p != 0:
            raise ValueError("MAE patchify requires square images divisible by patch size.")
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        if h * w != patches.shape[1]:
            raise ValueError("MAE unpatchify requires a square number of patches.")
        x = patches.reshape(shape=(patches.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(patches.shape[0], 3, h * p, h * p))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std + self.mean).clamp(0, 1)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))
        noise = torch.rand(n, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        mask = torch.ones([n, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def training_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = self.normalize_input(x)
        latent, mask, ids_restore = self.forward_encoder(model_input, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(model_input, pred, mask)
        image = self.denormalize_output(self.unpatchify(pred))
        return loss, image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"mae_vit_b_16 requires square {self.image_size}x{self.image_size} inputs, "
                f"got {w}x{h}."
            )
        mask_ratio = self.mask_ratio if self.training else 0.0
        model_input = self.normalize_input(x)
        latent, _, ids_restore = self.forward_encoder(model_input, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return self.denormalize_output(self.unpatchify(pred))


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
        return OfficialMaeVitRestoration(weights=weights, image_size=width)

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

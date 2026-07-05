from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from visuarecon.data import CleanPriorDataset, scan_clean_records, split_clean_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean-prior self-supervised reconstruction model.")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--model", default="fcn_resnet50", choices=[
        "fcn_resnet50",
        "deeplabv3_resnet50",
        "lraspp_mobilenet_v3_large",
        "mae_vit_b_16",
    ])
    parser.add_argument("--weights", default="default", choices=["default", "none"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 288], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--official-image-size",
        action="store_true",
        help="Use the torchvision official evaluation size for this model family.",
    )
    parser.add_argument("--degradation", default="blur_noise_erase", choices=["blur_noise_erase", "blur_noise_patchmask", "mild"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def effective_num_workers(num_workers: int) -> int:
    if os.name == "nt" and num_workers > 2:
        print(
            f"Windows DataLoader is using {num_workers} workers. If worker startup is slow, "
            "rerun with --num-workers 0 or 2."
        )
    return num_workers


def make_loader(dataset: CleanPriorDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, degradation: str) -> dict[str, float]:
    from visuarecon.degrade import degrade_batch
    from visuarecon.metrics import l1_loss, psnr

    model.eval()
    losses: list[float] = []
    psnrs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            clean = batch["image"].to(device, non_blocking=True)
            if hasattr(model, "training_loss"):
                pred = model(clean)
            else:
                degraded = degrade_batch(clean, mode=degradation)
                pred = model(degraded)
            losses.append(l1_loss(pred, clean).item())
            psnrs.append(psnr(pred, clean))
    return {
        "l1": float(np.mean(losses)) if losses else 0.0,
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
    }


def main() -> None:
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid

    from visuarecon.degrade import degrade_batch
    from visuarecon.metrics import l1_loss, mse_loss
    from visuarecon.models import ModelSpec, build_model, checkpoint_payload, official_image_size

    args = parse_args()
    set_seed(args.seed)
    image_size = official_image_size(args.model) if args.official_image_size else (int(args.image_size[0]), int(args.image_size[1]))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    clean_records = scan_clean_records(args.data_root)
    train_records, val_records = split_clean_records(clean_records, args.val_fraction, args.seed)
    if args.limit_train:
        train_records = train_records[: args.limit_train]
    if args.limit_val:
        val_records = val_records[: args.limit_val]
    if not train_records or not val_records:
        raise RuntimeError("Need at least one train and one validation clean image.")

    num_workers = effective_num_workers(args.num_workers)
    train_loader = make_loader(CleanPriorDataset(train_records, image_size), args.batch_size, True, num_workers)
    val_loader = make_loader(CleanPriorDataset(val_records, image_size), args.batch_size, False, num_workers)

    device = torch.device(args.device)
    model = build_model(args.model, args.weights, image_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    spec = ModelSpec(name=args.model, weights=args.weights, image_size=image_size)

    best_psnr = -float("inf")
    global_step = 0
    print(f"train clean images: {len(train_records)}; val clean images: {len(val_records)}")
    print(f"model={args.model} weights={args.weights} image_size={image_size} device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            clean = batch["image"].to(device, non_blocking=True)
            if hasattr(model, "training_loss"):
                loss, pred = model.training_loss(clean)
            else:
                degraded = degrade_batch(clean, mode=args.degradation)
                pred = model(degraded)
                loss = l1_loss(pred, clean) + 0.2 * mse_loss(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            writer.add_scalar("train/loss", loss_value, global_step)
            pbar.set_postfix(loss=f"{loss_value:.4f}")
            global_step += 1

        metrics = evaluate(model, val_loader, device, args.degradation)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("val/l1", metrics["l1"], epoch)
        writer.add_scalar("val/psnr", metrics["psnr"], epoch)

        with torch.no_grad():
            sample = next(iter(val_loader))["image"].to(device)
            if hasattr(model, "training_loss"):
                degraded = sample
                pred = model(sample)
            else:
                degraded = degrade_batch(sample, mode=args.degradation)
                pred = model(degraded)
            writer.add_image("val/degraded_clean", make_grid(degraded[:4].cpu(), nrow=4), epoch)
            writer.add_image("val/pred", make_grid(pred[:4].cpu(), nrow=4), epoch)
            writer.add_image("val/target", make_grid(sample[:4].cpu(), nrow=4), epoch)

        payload = checkpoint_payload(model, optimizer, epoch, best_psnr, spec)
        torch.save(payload, run_dir / "last.pt")
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            payload = checkpoint_payload(model, optimizer, epoch, best_psnr, spec)
            torch.save(payload, run_dir / "best.pt")

        print(
            f"epoch={epoch} train_loss={train_loss:.5f} "
            f"val_l1={metrics['l1']:.5f} val_psnr={metrics['psnr']:.2f} best_psnr={best_psnr:.2f}"
        )

    writer.close()


if __name__ == "__main__":
    main()

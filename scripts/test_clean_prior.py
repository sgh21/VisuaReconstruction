from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from visuarecon.data import CleanPriorDataset, SuctionDataset, scan_clean_records, scan_suction_records
from visuarecon.metrics import l1_loss, psnr
from visuarecon.models import build_model, official_image_size
from visuarecon.vis import append_csv, residual_mask, save_suction_visuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test clean-prior model and export suction residual mask overlays.")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--official-image-size",
        action="store_true",
        help="Use the torchvision official evaluation size for the checkpoint model family.",
    )
    parser.add_argument("--limit-clean", type=int, default=0)
    parser.add_argument("--limit-suction", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    return parser.parse_args()


def safe_stem(version: str, group: str, suffix: str) -> str:
    return f"{version}_{group}_{suffix}".replace("/", "_").replace("\\", "_")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(ckpt["model_name"])
    weights = str(ckpt.get("weights", "none"))
    if args.official_image_size:
        image_size = official_image_size(model_name)
    elif args.image_size:
        image_size = tuple(args.image_size)
    else:
        image_size = tuple(ckpt["image_size"])
    image_size = (int(image_size[0]), int(image_size[1]))

    device = torch.device(args.device)
    model = build_model(model_name, weights, image_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    clean_records = scan_clean_records(args.data_root)
    suction_records = scan_suction_records(args.data_root)
    if args.limit_clean:
        clean_records = clean_records[: args.limit_clean]
    if args.limit_suction:
        suction_records = suction_records[: args.limit_suction]

    clean_loader = DataLoader(
        CleanPriorDataset(clean_records, image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    suction_loader = DataLoader(
        SuctionDataset(suction_records, image_size),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    clean_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in tqdm(clean_loader, desc="clean eval"):
            clean = batch["image"].to(device)
            pred = model(clean)
            for i in range(clean.shape[0]):
                clean_rows.append(
                    {
                        "version": batch["version"][i],
                        "group": batch["group"][i],
                        "path": batch["path"][i],
                        "l1": float(l1_loss(pred[i : i + 1], clean[i : i + 1]).item()),
                        "psnr": float(psnr(pred[i : i + 1], clean[i : i + 1])),
                    }
                )

    suction_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in tqdm(suction_loader, desc="suction masks"):
            image = batch["image"].to(device)
            prior_hat = model(image)
            mask = residual_mask(image[0].cpu(), prior_hat[0].cpu())
            mask_mean = float(mask.mean().item())
            mask_p95 = float(torch.quantile(mask.flatten(), 0.95).item())
            version = batch["version"][0]
            group = batch["group"][0]
            index = int(batch["suction_index"][0])
            stem = output_dir / safe_stem(version, group, f"suction_{index:02d}")
            save_suction_visuals(image[0].cpu(), prior_hat[0].cpu(), mask, stem, alpha=args.overlay_alpha)
            suction_rows.append(
                {
                    "version": version,
                    "group": group,
                    "suction_index": index,
                    "path": batch["path"][0],
                    "mask_mean": mask_mean,
                    "mask_p95": mask_p95,
                }
            )

    append_csv(output_dir / "clean_metrics.csv", ["version", "group", "path", "l1", "psnr"], clean_rows)
    append_csv(
        output_dir / "suction_mask_stats.csv",
        ["version", "group", "suction_index", "path", "mask_mean", "mask_p95"],
        suction_rows,
    )
    if clean_rows:
        mean_l1 = float(np.mean([r["l1"] for r in clean_rows]))
        mean_psnr = float(np.mean([r["psnr"] for r in clean_rows]))
        print(f"clean mean_l1={mean_l1:.5f} mean_psnr={mean_psnr:.2f}")
    print(f"saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()

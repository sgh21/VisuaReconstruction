from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CleanRecord:
    version: str
    group: str
    clean_path: Path


@dataclass(frozen=True)
class SuctionRecord:
    version: str
    group: str
    index: int
    suction_path: Path


def scan_clean_records(data_root: str | Path) -> list[CleanRecord]:
    root = Path(data_root)
    records: list[CleanRecord] = []
    for version_dir in sorted(root.glob("dataset_v*")):
        if not version_dir.is_dir():
            continue
        for group_dir in sorted(version_dir.glob("group_*")):
            clean_path = group_dir / "clean.png"
            if clean_path.is_file():
                records.append(
                    CleanRecord(
                        version=version_dir.name,
                        group=group_dir.name,
                        clean_path=clean_path,
                    )
                )
    return records


def scan_suction_records(data_root: str | Path) -> list[SuctionRecord]:
    root = Path(data_root)
    records: list[SuctionRecord] = []
    for version_dir in sorted(root.glob("dataset_v*")):
        if not version_dir.is_dir():
            continue
        for group_dir in sorted(version_dir.glob("group_*")):
            for suction_path in sorted(group_dir.glob("suction_*.png")):
                stem = suction_path.stem
                try:
                    index = int(stem.split("_")[-1])
                except ValueError:
                    index = -1
                records.append(
                    SuctionRecord(
                        version=version_dir.name,
                        group=group_dir.name,
                        index=index,
                        suction_path=suction_path,
                    )
                )
    return records


def split_clean_records(
    records: Iterable[CleanRecord],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[CleanRecord], list[CleanRecord]]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
    val_records = sorted(items[:val_count], key=lambda r: (r.version, r.group))
    train_records = sorted(items[val_count:], key=lambda r: (r.version, r.group))
    return train_records, val_records


def center_crop_exact(img: Image.Image, crop_size: int = 1080) -> Image.Image:
    width, height = img.size
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"Image {width}x{height} is smaller than required center crop {crop_size}x{crop_size}."
        )
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return img.crop((left, top, left + crop_size, top + crop_size))


def load_rgb_tensor(path: str | Path, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = image_size
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = center_crop_exact(img, crop_size=1080)
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CleanPriorDataset(Dataset):
    def __init__(self, records: list[CleanRecord], image_size: tuple[int, int]) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = load_rgb_tensor(record.clean_path, self.image_size)
        return {
            "image": image,
            "version": record.version,
            "group": record.group,
            "path": str(record.clean_path),
        }


class SuctionDataset(Dataset):
    def __init__(self, records: list[SuctionRecord], image_size: tuple[int, int]) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = load_rgb_tensor(record.suction_path, self.image_size)
        return {
            "image": image,
            "version": record.version,
            "group": record.group,
            "suction_index": record.index,
            "path": str(record.suction_path),
        }

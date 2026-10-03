"""Общие функции проекта."""
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["non_food", "food"]  # индекс = метка класса (0 — не еда, 1 — еда)
SPLITS = {"train": "training", "val": "validation", "test": "evaluation"}


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_from_name(path: Path) -> int:
    """В Food-5K метка закодирована префиксом имени файла: 1_*.jpg — еда, 0_*.jpg — не еда."""
    return int(path.name.split("_", 1)[0])


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_npz(path):
    d = np.load(path)
    return d["X"], d["y"]

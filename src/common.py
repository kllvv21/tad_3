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
    """Метка класса: 1 — еда, 0 — не еда.

    Поддерживаются обе раскладки датасета Food-5K:
      * имя файла с префиксом метки: 1_123.jpg / 0_45.jpg (оригинальный архив EPFL);
      * каталог класса: .../food/123.jpg, .../non_food/45.jpg (версия на Kaggle).
    """
    for part in reversed(path.parts[:-1]):
        name = part.lower().replace("-", "_")
        if name in ("non_food", "nonfood"):
            return 0
        if name == "food":
            return 1
    prefix = path.name.split("_", 1)[0]
    if "_" in path.name and prefix in ("0", "1"):
        return int(prefix)
    raise ValueError(f"Не удалось определить класс изображения: {path}")


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

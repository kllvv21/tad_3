"""Стадия prepare: чтение исходных изображений Food-5K, приведение к единому размеру,
сохранение выборок train/val/test в сжатом виде (numpy .npz)."""
import os

import numpy as np
from PIL import Image, ImageOps

from common import ROOT, SPLITS, label_from_name, list_images, load_params, save_json

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prepared"


def load_image(path, size):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # центрированная обрезка до квадрата с сохранением пропорций и масштабирование
    img = ImageOps.fit(img, (size, size), method=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def main():
    p = load_params()["prepare"]
    size = p["img_size"]
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for split, folder in SPLITS.items():
        files = list_images(RAW / folder)
        X = np.stack([load_image(f, size) for f in files])
        y = np.array([label_from_name(f) for f in files], dtype=np.int64)
        np.savez_compressed(OUT / f"{split}.npz", X=X, y=y)
        stats[split] = {"n": int(len(y)), "food": int(y.sum()), "non_food": int((1 - y).sum())}
        print(f"{split}: {X.shape}, food={y.sum()}, non_food={(1 - y).sum()}")
    save_json(stats, str(ROOT / "reports" / "data_stats.json"))


if __name__ == "__main__":
    main()

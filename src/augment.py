"""Стадия augment: генерация дополнительных обучающих изображений с помощью
случайных преобразований и сохранение нового датасета в data/augmented/<класс>/*.jpg.

Используемые преобразования (применяются в случайной комбинации):
  * горизонтальное отражение;
  * поворот на случайный угол в диапазоне [-max_rotation; max_rotation];
  * случайная обрезка (random resized crop) с последующим масштабированием;
  * изменение яркости, контраста и насыщенности;
  * гауссовский шум;
  * гауссовское размытие (с вероятностью blur_prob).
Аугментируется только обучающая выборка — валидационная и тестовая остаются исходными.
"""
import os
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from common import CLASSES, ROOT, load_npz, load_params, save_json

SRC = ROOT / "data" / "prepared" / "train.npz"
OUT = ROOT / "data" / "augmented"


def augment_one(img: Image.Image, rng: np.random.Generator, p: dict) -> Image.Image:
    w, h = img.size
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-p["max_rotation"], p["max_rotation"])
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=None)
    # поворот оставляет чёрные углы — убираем их случайной обрезкой
    scale = rng.uniform(p["crop_scale_min"], 1.0)
    cw, ch = int(w * np.sqrt(scale)), int(h * np.sqrt(scale))
    x0 = rng.integers(0, w - cw + 1)
    y0 = rng.integers(0, h - ch + 1)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-p["brightness"], p["brightness"]))
    img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-p["contrast"], p["contrast"]))
    img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-p["saturation"], p["saturation"]))
    if rng.random() < p["blur_prob"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
    if p["noise_std"] > 0:
        arr = np.asarray(img, dtype=np.float32)
        arr += rng.normal(0, p["noise_std"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def main():
    params = load_params()
    p = params["augment"]
    rng = np.random.default_rng(params["base"]["seed"])
    X, y = load_npz(SRC)
    if OUT.exists():
        shutil.rmtree(OUT)
    for c in CLASSES:
        os.makedirs(OUT / c, exist_ok=True)
    n = 0
    for i, (arr, label) in enumerate(zip(X, y)):
        img = Image.fromarray(arr)
        for k in range(p["copies"]):
            aug = augment_one(img, rng, p)
            # имя сохраняет формат Food-5K: <метка>_<номер>.jpg
            aug.save(OUT / CLASSES[label] / f"{label}_aug{i:04d}_{k}.jpg", quality=95)
            n += 1
    print(f"Сгенерировано {n} аугментированных изображений из {len(y)} исходных")
    save_json({"source": int(len(y)), "generated": n, "copies": p["copies"]},
              str(ROOT / "reports" / "augment_stats.json"))


if __name__ == "__main__":
    main()

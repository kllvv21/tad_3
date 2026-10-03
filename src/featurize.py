"""Стадия featurize: извлечение признаков из изображений.

Вектор признаков изображения состоит из:
  * HOG (гистограммы направленных градиентов) — форма и контуры объектов;
  * 3D-гистограммы цвета в пространстве HSV — цветовая гамма (еда обычно «тёплых» оттенков);
  * гистограммы локальных бинарных шаблонов (LBP) — текстура;
  * цветовых моментов (среднее и СКО по каналам RGB и HSV).
"""
import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from common import ROOT, label_from_name, list_images, load_npz, load_params

PREP = ROOT / "data" / "prepared"
AUG = ROOT / "data" / "augmented"
OUT = ROOT / "data" / "features"


def extract(img: np.ndarray, p: dict) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f_hog = hog(gray, orientations=p["hog_orientations"],
                pixels_per_cell=(p["hog_cell"], p["hog_cell"]),
                cells_per_block=(p["hog_block"], p["hog_block"]),
                block_norm="L2-Hys", feature_vector=True)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    b = p["color_bins"]
    f_hist = cv2.calcHist([hsv], [0, 1, 2], None, [b, b, b], [0, 180, 0, 256, 0, 256]).flatten()
    f_hist /= f_hist.sum() + 1e-9
    P, R = p["lbp_points"], p["lbp_radius"]
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    f_lbp, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
    moments = np.concatenate([img.reshape(-1, 3).mean(0), img.reshape(-1, 3).std(0),
                              hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)]) / 255.0
    return np.concatenate([f_hog, f_hist, f_lbp, moments]).astype(np.float32)


def featurize(images, p):
    feats = Parallel(n_jobs=-1, batch_size=64)(delayed(extract)(im, p) for im in images)
    return np.stack(feats)


def main():
    p = load_params()["featurize"]
    os.makedirs(OUT, exist_ok=True)
    for split in ["train", "val", "test"]:
        X, y = load_npz(PREP / f"{split}.npz")
        F = featurize(X, p)
        np.savez_compressed(OUT / f"{split}.npz", X=F, y=y)
        print(f"{split}: признаки {F.shape}")
    files = list_images(AUG)
    imgs = (np.asarray(Image.open(f).convert("RGB")) for f in files)
    F = featurize(imgs, p)
    y = np.array([label_from_name(f) for f in files], dtype=np.int64)
    np.savez_compressed(OUT / "aug.npz", X=F, y=y)
    print(f"aug: признаки {F.shape}")


if __name__ == "__main__":
    main()

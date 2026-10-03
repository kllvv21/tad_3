"""Стадия train: обучение выбранной модели (параметр train.model)
на исходных или исходных + аугментированных данных (параметр train.use_augmented)."""
import os
import pickle
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import ROOT, load_npz, load_params, save_json

FEAT = ROOT / "data" / "features"
MODEL = ROOT / "models" / "model.pkl"


def build_model(name: str, mp: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(C=mp["C"], max_iter=mp["max_iter"], random_state=seed)
    if name == "svm":
        return SVC(C=mp["C"], gamma=mp["gamma"], kernel="rbf", probability=True, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=mp["n_estimators"],
                                      max_depth=mp["max_depth"] or None,
                                      min_samples_leaf=mp["min_samples_leaf"],
                                      n_jobs=-1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=mp["learning_rate"],
                                              max_iter=mp["max_iter"],
                                              max_leaf_nodes=mp["max_leaf_nodes"],
                                              random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=mp["n_neighbors"], weights=mp["weights"], n_jobs=-1)
    raise ValueError(f"Неизвестная модель: {name}")


def main():
    params = load_params()
    tp, seed = params["train"], params["base"]["seed"]
    name = tp["model"]
    X, y = load_npz(FEAT / "train.npz")
    if tp["use_augmented"]:
        Xa, ya = load_npz(FEAT / "aug.npz")
        X, y = np.concatenate([X, Xa]), np.concatenate([y, ya])
    steps = [("scaler", StandardScaler())]
    if tp["pca_components"]:
        steps.append(("pca", PCA(n_components=tp["pca_components"], random_state=seed)))
    steps.append(("clf", build_model(name, params["models"][name], seed)))
    pipe = Pipeline(steps)
    t0 = time.time()
    pipe.fit(X, y)
    train_time = time.time() - t0
    os.makedirs(MODEL.parent, exist_ok=True)
    with open(MODEL, "wb") as f:
        pickle.dump(pipe, f)
    info = {"model": name, "use_augmented": tp["use_augmented"],
            "n_train": int(len(y)), "n_features": int(X.shape[1]),
            "train_time_sec": round(train_time, 2)}
    save_json(info, str(ROOT / "reports" / "train_info.json"))
    print(info)


if __name__ == "__main__":
    main()

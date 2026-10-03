"""Стадия evaluate: оценка качества обученной модели на валидационной и тестовой
выборках; сохранение метрик (metrics.json) и данных для графиков DVC."""
import json
import os
import pickle

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from common import CLASSES, ROOT, load_npz, save_json

FEAT = ROOT / "data" / "features"
MODEL = ROOT / "models" / "model.pkl"
REP = ROOT / "reports"


def scores(y, pred, proba):
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, proba), 4)}


def main():
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    with open(REP / "train_info.json", encoding="utf-8") as f:
        info = json.load(f)
    metrics = {}
    for split in ["val", "test"]:
        X, y = load_npz(FEAT / f"{split}.npz")
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics[split] = scores(y, pred, proba)
        if split == "test":
            os.makedirs(REP / "plots", exist_ok=True)
            # данные для dvc plots: матрица ошибок и ROC-кривая
            save_json([{"actual": CLASSES[a], "predicted": CLASSES[p]} for a, p in zip(y, pred)],
                      str(REP / "plots" / "confusion.json"))
            fpr, tpr, _ = roc_curve(y, proba)
            idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)
            save_json([{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
                      str(REP / "plots" / "roc.json"))
            metrics["test"]["confusion_matrix"] = confusion_matrix(y, pred).tolist()
    metrics["train_time_sec"] = info["train_time_sec"]
    metrics["n_train"] = info["n_train"]
    save_json(metrics, str(REP / "metrics.json"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

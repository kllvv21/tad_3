#!/usr/bin/env bash
# Серия экспериментов: 5 моделей x 2 варианта данных (без аугментации / с аугментацией)
set -e
for aug in false true; do
  for m in logreg svm rf hgb knn; do
    if [ "$aug" = "true" ]; then name="$m-aug"; else name="$m-base"; fi
    dvc exp run -n "$name" -S train.model=$m -S train.use_augmented=$aug
  done
done
dvc exp show --no-pager

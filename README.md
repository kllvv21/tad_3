# Food vs Non-Food: MLOps-пайплайн на DVC

Практическая работа № 3 «Инженерия машинного обучения» (дисциплина «Технологии анализа данных», СФУ ИКИТ).

Задача — бинарная классификация изображений «еда / не еда» на датасете **Food-5K**
(Food vs Non-Food Image Dataset): 3000 обучающих, 1000 валидационных и 1000 тестовых изображений,
классы сбалансированы.

## Структура

```
data/raw/            исходный датасет (под контролем DVC, data/raw.dvc)
data/prepared/       изображения 128x128 в формате .npz           (стадия prepare)
data/augmented/      новый аугментированный датасет (.jpg)        (стадия augment)
data/features/       векторы признаков HOG + HSV + LBP            (стадия featurize)
models/model.pkl     обученная модель                              (стадия train)
reports/             метрики и данные для графиков                 (стадия evaluate)
src/                 код стадий
params.yaml          все гиперпараметры
dvc.yaml / dvc.lock  описание и состояние пайплайна
run_experiments.sh   серия из 10 экспериментов
```

Пайплайн: `prepare → augment → featurize → train → evaluate`.

## Воспроизведение

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# датасет: https://www.kaggle.com/datasets/trolukovich/food5k-image-dataset
# распаковать так, чтобы получилось data/raw/{training,validation,evaluation}/...
# (метка берётся из префикса имени файла: 1_*.jpg — еда, 0_*.jpg — не еда)

dvc repro                 # прогон всего пайплайна
./run_experiments.sh      # 5 моделей x {без аугментации, с аугментацией}
```

## Работа с экспериментами

```bash
dvc exp show                                    # таблица всех экспериментов
dvc exp show --only-changed --sort-by reports/metrics.json:test.accuracy --sort-order desc
dvc exp diff hgb-base hgb-aug                   # сравнение двух экспериментов
dvc exp apply hgb-aug                           # перейти к выбранному эксперименту
dvc exp branch hgb-aug best-model               # сохранить эксперимент в отдельную ветку
dvc plots diff hgb-base hgb-aug                 # графики ROC и матрицы ошибок
dvc exp run -S train.model=svm -S models.svm.C=10   # новый эксперимент
```

Эксперименты хранятся в git-ссылках `refs/exps/...`; чтобы они попали на GitHub:
`dvc exp push origin --all` (или `git push origin 'refs/exps/*:refs/exps/*'`), получить — `dvc exp pull origin --all`.

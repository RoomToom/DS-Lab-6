# Лабораторна робота №6 — Кластеризація та PCA

Аналіз червоних і білих вин із набору UCI Wine Quality.

У `src/tasks.py` виконуються об’єднання даних, видалення неповних рядків, фільтрація викидів за правилом 1.5 IQR, one-hot кодування типу вина, стандартизація, PCA до двох компонентів і K-means із трьома кластерами.
`quality` не входить до ознак кластеризації. Для K-means встановлено `random_state=42` і `n_init=10`.

## Встановлення

Python 3.11. Команди виконуються з кореня репозиторію.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

У Linux/macOS: `source venv/bin/activate`.

## Запуск

```powershell
python -m src.tasks
```

Результати в `results/`:

- `clusters.png` — кластери у просторі двох головних компонентів;
- `wine_clusters.csv` — очищені дані, координати PCA й номери кластерів;
- `summary.json` — кількість рядків до та після очищення, розміри кластерів і коефіцієнт силуету.

Номери кластерів позначають групи схожих вин, а не оцінку їхньої якості.

## Перевірка

```powershell
python -m pytest tests/ -v
```

[Умова лабораторної](https://github.com/UNI-DS/DS-Lab-6) · [Джерело даних](https://archive.ics.uci.edu/dataset/186/wine+quality)

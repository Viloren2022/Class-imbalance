# Программные материалы НИР-3

Тема: Исследование устойчивости методов обработки дисбаланса классов при прогнозировании отказов промышленного оборудования на синтетических и реальных данных.

## Состав

- `data/`: исходные UCI CSV (AI4I 2020, Scania train/test), а также AI4I-PMDI. Исходные файлы сохраняются без ручного редактирования.
- `chapter2_experiment.py`: подготовка данных, исключение признаков с утечкой, создание конвейеров, стратегии балансировки, показатели классификации.
- `run_unified_24_cv.py`: 24 конфигурации, 5-fold CV, финальное обучение и независимый test; сохраняет индивидуальные вероятности, OOF-пороги, CSV и manifest.
- `bootstrap_unified.py`: стратифицированный парный bootstrap (по умолчанию B=5000) для S, двух контрастов и стоимости ошибок на фиксированных тестовых вероятностях.
- `bootstrap_AP_1000.py`: 95%-ные интервалы AP для 24 конфигураций и 18 парных разностей ΔAP (по умолчанию B=1000).
- `aggregate_measures.py`: вычисление ΔAP, G, D, S и двух альтернативных нормировок по тестовой таблице.
- `describe_data.py`: формирование характеристик четырех обучающих/тестовых частей из исходных CSV.
- `verify_saved_results.py`: независимый контроль 24 AP, FP/FN и стоимости по `test_predictions.npz` без повторного обучения.
- `Results_Review.ipynb`: просмотр основных результатов и проверка сохраненных прогнозов.
- `results/`: готовые CSV, файл индивидуальных вероятностей, параметры среды и SHA-256, интервалы bootstrap.
- `results/historical_primary/`: отдельная историческая серия, сохраненная для проверки ранее заявленной H1, с другими настройками; ее нельзя сливать с унифицированной.
- `PMDI_Control/`: отдельный ранее выполненный эксперимент на AI4I-PMDI, две целевые разметки, контроль случайных пропусков, интервалы и программы; серия не входит в H1.

## Среда выполнения

Обучение проводилось в Python 3.13.5 со scikit-learn 1.8.0, imbalanced-learn 0.14.1, XGBoost 3.1.3, NumPy 2.3.5, pandas 2.2.3. Параметры зафиксированы в `results/manifest.json`, зависимости – в `requirements.txt`.

```bash
python -m venv .venv
# Активировать виртуальное окружение и затем:
python -m pip install -r requirements.txt
```

## Контроль полученных результатов без обучения

Из каталога `Source_Code`:

```bash
python verify_saved_results.py
python aggregate_measures.py --results-dir results --out-dir results/recomputed
python describe_data.py --data-dir data --out-dir results/recomputed
python bootstrap_AP_1000.py --results-dir results --out-dir results/recomputed --replicates 1000
python bootstrap_unified.py --results-dir results --out-dir results/recomputed --replicates 5000
```

`bootstrap_unified.py` рассчитан на полный комплект из 24 конфигураций. Для выбранных методов и набора данных используется одна и та же повторная выборка тестовых объектов (парность). Число положительных и отрицательных объектов сохраняется. Модели не переобучаются внутри bootstrap-повторов; интервал относится к фиксированным прогнозам и не включает вариацию их обучения.

## Полное повторное обучение

```bash
python run_unified_24_cv.py --data-dir data --out-dir results/recomputed_full
python verify_saved_results.py --results-dir results/recomputed_full
python bootstrap_unified.py --results-dir results/recomputed_full --out-dir results/recomputed_full --replicates 5000
python aggregate_measures.py --results-dir results/recomputed_full --out-dir results/recomputed_full
python bootstrap_AP_1000.py --results-dir results/recomputed_full --out-dir results/recomputed_full --replicates 1000
```

Для быстрой проверки исполняемости одного конвейера:

```bash
python run_unified_24_cv.py --only-dataset AI4I --only-model logreg --only-treatment baseline --out-dir results/smoke
```

Результаты такого сокращенного запуска не являются полной серией и не подходят для `bootstrap_unified.py`.

### Постановка

- AI4I: train/test 80/20, стратификация, random_state=42; используются `Type` и 5 измерений. UDI, Product ID и TWF/HDF/PWF/OSF/RNF исключены.
- Scania: исходное train/test UCI; 170 числовых признаков, `na` – пропущенное значение.
- Ресемплинг/импутация/масштабирование подбираются только внутри train/fold; на test не обучаются.
- Модели: Logistic Regression (lbfgs), Random Forest (50 деревьев, max_samples=0.5), XGBoost (400 деревьев в базовом скрипте); baseline, weighting, ROS, SMOTE; AI4I использует SMOTENC для смешанных признаков.
- Average Precision – основная метрика; Precision, Recall, F1 и матрицы ошибок при 0,5; для Scania стоимость C = 10·FP + 500·FN при пороге, выбранном исключительно по OOF train.
- Проверочная унифицированная серия не переопределяет исходное решение по заранее согласованной H1.

## Важное разграничение серий

`results/test_and_cv_summary.csv`, `results/unified_bootstrap_intervals_5000.csv` и `results/test_predictions.npz` относятся к ОДНОМУ унифицированному расчету. Исторические AP и H1 размещены отдельно в `historical_primary`. AI4I-PMDI – контрольная серия со своей конфигурацией моделей (в том числе RF 400 деревьев и LR liblinear) и собственной меткой `Diagnostic`. Смешивание результатов этих серий в одной таблице без идентификатора метода дает несопоставимые оценки.

## Выходные показатели

- `test_metrics_complete.csv`: 24 строки, AP, Precision, Recall, F1, TN/FP/FN/TP и OOF-пороги.
- `cv_5fold_metrics.csv`: 120 строк по проверочным фолдам.
- `AP_test_CI_1000.csv`: 24 95%-ных интервала AP по фиксированным прогнозам.
- `delta_AP_test_CI_1000.csv`: 18 парных 95%-ных интервалов ΔAP.
- `normalized_effects_G_D.csv`, `stability_summary_S.csv`: межконтекстные эффекты по различным нормировкам.
- `unified_bootstrap_5000.csv`, `unified_bootstrap_intervals_5000.csv`: распределение 5000 повторов и интервалы S/стоимости.
- `dataset_overview.csv`: данные четырех частей.
- `verification_manifest.json`: контрольные проверки и SHA-256 исходных файлов.

## Источники

AI4I 2020: https://archive.ics.uci.edu/dataset/601/ (DOI 10.24432/C5HS5C).
Scania APS: https://archive.ics.uci.edu/dataset/421/ (DOI 10.24432/C51S51).
AI4I-PMDI: https://figshare.com/articles/dataset/24718428 (DOI 10.6084/m9.figshare.24718428.v1).

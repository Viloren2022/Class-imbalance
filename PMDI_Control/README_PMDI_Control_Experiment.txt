Дополнительная контрольная серия AI4I-PMDI для главы 2
Версия: 2026-11-02

Состав пакета
- Chapter2_PMDI_Control_Experiment.docx: академический отчет с методикой, пятью таблицами, выводами и источниками.
- ai4i_pmdi_experiment.py: 5-fold CV и test по 12 конфигурациям в двух вариантах целевой переменной.
- ai4i_pmdi_robustness.py: повторное обучение исходного AI4I с совпадающими настройками, парный стратифицированный bootstrap 1000 и проверка хронологического разделения.
- ch2_pmdi/exploratory_masking.py: три контрольные реализации независимого случайного удаления измерений (MCAR).
- ch2_pmdi/reconcile_outputs.py: пересборка сравнительных таблиц из сопоставимых расчетов.
- ch2_pmdi/xgb_within_bootstrap.py: дополнительный парный bootstrap XGBoost с ROS/SMOTE относительно baseline на PMDI.
- ch2_pmdi/*.csv и ch2_pmdi/*.json: исходные численные результаты и аудит; *.npz: вероятности положительного класса для парных проверок.
- ai4i2020.csv и AI4I-PMDI.csv: исходные данные контрольной серии.
- chapter2_experiment.py: функции базового эксперимента для воспроизведения сопоставимых моделей.
- ch2_results_final/test_summary.csv: исходная сводка основной серии для проверки различий конфигурации.

Методика и важное предупреждение
Контрольная серия включает 24 конфигурации (12 на каждом варианте целевой переменной), 5-fold CV, независимый test (20%), Random State 42.
Сопоставление по UDI, train/test совпадают между наборами.
Контрольная серия использует логистическую регрессию liblinear, RF 400 деревьев, XGBoost 400 итераций.
В ОСНОВНОЙ серии главы 2 RF ранее был упрощен до 50 деревьев и другого параметра подвыборки, а логистическая регрессия имела другой решатель. Поэтому для PMDI создан повторный проход исходного AI4I с ИДЕНТИЧНЫМИ параметрами; для анализа парного эффекта берите только ch2_pmdi/paired_AP_verified.csv и comparison_original_vs_pmdi.csv, а не старые числа из основной серии. Решение по H1 не меняется.
Диагностическая переменная PMDI имеет 348 положительных меток, исходная Machine failure - 339; расходятся 27 строк. Чистое сопоставление полноты данных - matched_labels, а native_diagnostic - альтернативный исход.
Пропуски в PMDI структурированы переменной Control и связаны с видами отказов, поэтому результаты не эквивалентны эксперименту с произвольным удалением ячеек.

Запуск (Python >= 3.11)
Установить зависимости из requirements_chapter2.txt. Распаковать архив целиком в одну директорию, далее выполнить из этой директории:
python ai4i_pmdi_experiment.py
python ai4i_pmdi_robustness.py
python ch2_pmdi/exploratory_masking.py
python ch2_pmdi/reconcile_outputs.py
python ch2_pmdi/xgb_within_bootstrap.py

Примечание: файлы результатов уже включены; повторный запуск необязателен. Оценки bootstrap отличаются при изменении seed или версии библиотек.

Научные источники
Autran J.-V. AI4I-PMDI: Predictive Maintenance Dataset with Irregularities. Figshare (2024). DOI: 10.6084/m9.figshare.24718428.v1.
Autran et al. (2024). Procedia Computer Science 246: 1201-1209. DOI: 10.1016/j.procs.2024.09.546.
UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/601.

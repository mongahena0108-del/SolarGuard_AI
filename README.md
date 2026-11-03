# SolarGuard AI

Streamlit MVP for solar inverter performance monitoring, expected-output estimation, and daylight anomaly prioritisation.

## Launch

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

The app recursively locates the four required Kaggle CSV files below the project directory. It parses inconsistent timestamps, deduplicates plant-level weather by `PLANT_ID` and `DATE_TIME`, and merges it to every inverter record without inventing inverter-specific weather.

## Models

- Random Forest Regressor estimates expected `AC_POWER` using operating sensors, time features, and one-hot inverter identity. It uses a chronological 80/20 split and reports MAE, RMSE, and R² on the held-out tail.
- Isolation Forest scores unusual daylight operating patterns using power, environmental sensors, and a safe AC/DC ratio. Nighttime zero generation is excluded from anomaly fitting.

## Rigorous validation

Run `python validation.py` to perform a fresh chronological holdout evaluation. The script trains Version A with `DC_POWER` and Version B without it, evaluates only later records with measured irradiation above `0.02`, selects 40 genuine cases across 40 inverter IDs, and writes:

- `model_evaluation_report.txt`
- `model_comparison.csv`
- `40_case_predictions.csv`
- `heldout_daylight_predictions.csv`
- `isolation_forest_test_summary.csv`
- `actual_vs_predicted.html`
- `error_distribution.html`

The dashboard's **Model Validation** page displays these held-out results separately from the original model-performance page. Version A's high score must be qualified because contemporaneous `DC_POWER` is closely related to `AC_POWER`; Version B is the more conservative environmental generalization check.

Run `python investigation.py` after validation to investigate the Isolation Forest boundary. It writes `isolation_forest_investigation.txt`, `contamination_comparison.csv`, `inverter_anomaly_summary.csv`, `plant_anomaly_summary.csv`, `hourly_anomaly_summary.csv`, `anomaly_feature_analysis.csv`, `anomaly_deviation_groups.csv`, controlled feature-set comparisons, and Plotly HTML charts. The Model Validation page displays the investigation without changing the original status logic.

## Limitations

There are no ground-truth fault labels, so no fault-detection accuracy is claimed. Expected power is a statistical estimate, not a guaranteed physical maximum. Weather readings are plant-level, not inverter-specific. Status thresholds are provisional prioritisation rules and an anomaly is not proof of failure; physical verification is required.

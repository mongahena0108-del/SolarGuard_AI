"""Unseen-data validation for SolarGuard AI.

This module intentionally does not modify source CSVs. It performs a fresh
chronological evaluation and writes judge-facing artifacts to this folder.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT
GEN_COLUMNS = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"}
WEATHER_COLUMNS = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"}
NUMERIC = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
BASE_FEATURES = ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HOUR", "DAY", "HOUR_SIN", "HOUR_COS", "SOURCE_KEY"]
FEATURES_A = ["DC_POWER"] + BASE_FEATURES
FEATURES_B = BASE_FEATURES


def load_validation_data():
    frames = []
    for plant_num in (1, 2):
        gen_path = next(ROOT.rglob(f"Plant_{plant_num}_Generation_Data.csv"), None)
        weather_path = next(ROOT.rglob(f"Plant_{plant_num}_Weather_Sensor_Data.csv"), None)
        if gen_path is None or weather_path is None:
            raise FileNotFoundError(f"Could not find both files for Plant {plant_num}")
        generation = pd.read_csv(gen_path)
        weather = pd.read_csv(weather_path)
        if not GEN_COLUMNS.issubset(generation.columns) or not WEATHER_COLUMNS.issubset(weather.columns):
            raise ValueError(f"Unexpected schema for Plant {plant_num}")
        generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], format="mixed", dayfirst=True, errors="coerce")
        weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format="mixed", errors="coerce")
        weather = weather.dropna(subset=["PLANT_ID", "DATE_TIME"])
        weather = weather.groupby(["PLANT_ID", "DATE_TIME"], as_index=False)[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean()
        merged = generation.merge(weather, on=["PLANT_ID", "DATE_TIME"], how="left", validate="many_to_one")
        merged["PLANT"] = f"Plant {plant_num}"
        frames.append(merged)
    data = pd.concat(frames, ignore_index=True)
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"], errors="coerce")
    data = data.dropna(subset=["DATE_TIME", "SOURCE_KEY", "AC_POWER", "DC_POWER"])
    data[NUMERIC] = data[NUMERIC].fillna(0)
    data["HOUR"] = data.DATE_TIME.dt.hour + data.DATE_TIME.dt.minute / 60
    data["DAY"] = data.DATE_TIME.dt.day_of_year
    data["HOUR_SIN"] = np.sin(2 * np.pi * data.HOUR / 24)
    data["HOUR_COS"] = np.cos(2 * np.pi * data.HOUR / 24)
    data["AC_DC_RATIO"] = np.where(data.DC_POWER > 0, data.AC_POWER / data.DC_POWER, 0).clip(0, 2)
    # Use measured plant irradiation for validation; AC power alone can be a
    # misleading daylight proxy when evaluating unusual operating records.
    data["DAYLIGHT"] = data.IRRADIATION > 0.02
    return data.sort_values("DATE_TIME").reset_index(drop=True)


def make_model(features):
    transformer = ColumnTransformer([("inverter", OneHotEncoder(handle_unknown="ignore"), ["SOURCE_KEY"])], remainder="passthrough")
    return Pipeline([("encode", transformer), ("forest", RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=42, n_jobs=-1))])


def metric_row(model_name, actual, predicted, scope):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    nonzero = np.abs(actual) > 1e-9
    return {"model": model_name, "scope": scope, "records": len(actual), "MAE": mean_absolute_error(actual, predicted), "RMSE": mean_squared_error(actual, predicted) ** 0.5, "R2": r2_score(actual, predicted) if len(np.unique(actual)) > 1 else np.nan, "MAPE_nonzero_pct": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else np.nan}


def choose_40_cases(test, predictions_a, predictions_b):
    candidate = test[test.DAYLIGHT].copy()
    candidate["_a"] = predictions_a.loc[candidate.index]
    candidate["_b"] = predictions_b.loc[candidate.index]
    candidate["irradiation_band"] = pd.qcut(candidate.IRRADIATION.rank(method="first"), 4, labels=False)
    # First select one varied daylight case per inverter, then fill remaining
    # slots by plant and irradiation band. This avoids a single easy regime.
    selected = []
    for inverter, group in candidate.groupby("SOURCE_KEY", sort=True):
        band = int(group.irradiation_band.iloc[len(group) // 2])
        selected.append(group[group.irradiation_band == band].iloc[len(group[group.irradiation_band == band]) // 2])
    chosen = pd.DataFrame(selected).drop_duplicates(subset=["PLANT", "SOURCE_KEY"])
    remaining = candidate.drop(index=chosen.index, errors="ignore")
    if len(chosen) < 40:
        extra = remaining.groupby(["PLANT", "irradiation_band"], group_keys=False).apply(lambda g: g.sample(min(len(g), 40), random_state=42), include_groups=False)
        chosen = pd.concat([chosen, extra]).drop_duplicates().head(40)
    if len(chosen) > 40:
        chosen = chosen.sample(40, random_state=42)
    chosen = chosen.sort_values(["PLANT", "SOURCE_KEY", "DATE_TIME"]).copy()
    chosen["Predicted_AC_POWER"] = chosen["_a"]
    chosen["Predicted_AC_POWER_Model_A"] = chosen["_a"]
    chosen["Predicted_AC_POWER_Model_B"] = chosen["_b"]
    chosen["Absolute_Error"] = np.abs(chosen.AC_POWER - chosen.Predicted_AC_POWER)
    chosen["Percentage_Error"] = np.where(np.abs(chosen.AC_POWER) > 1e-9, chosen.Absolute_Error / np.abs(chosen.AC_POWER) * 100, np.nan)
    chosen.insert(0, "Test_Case", range(1, len(chosen) + 1))
    return chosen


def evaluate(data=None):
    if data is None:
        data = load_validation_data()
    cutoff = data.DATE_TIME.quantile(0.8)
    train = data[data.DATE_TIME <= cutoff].copy()
    test = data[data.DATE_TIME > cutoff].copy()
    train_daylight = train[train.DAYLIGHT]
    test_daylight = test[test.DAYLIGHT]
    model_a = make_model(FEATURES_A)
    model_b = make_model(FEATURES_B)
    model_a.fit(train[FEATURES_A], train.AC_POWER)
    model_b.fit(train[FEATURES_B], train.AC_POWER)
    pred_a = pd.Series(model_a.predict(test[FEATURES_A]), index=test.index)
    pred_b = pd.Series(model_b.predict(test[FEATURES_B]), index=test.index)
    comparison = pd.DataFrame([metric_row("Version A - with DC_POWER", test_daylight.AC_POWER, pred_a.loc[test_daylight.index], "complete held-out daylight test"), metric_row("Version B - environmental only", test_daylight.AC_POWER, pred_b.loc[test_daylight.index], "complete held-out daylight test")])
    cases = choose_40_cases(test, pred_a, pred_b)
    case_metrics = pd.DataFrame([metric_row("Version A - with DC_POWER", cases.AC_POWER, cases.Predicted_AC_POWER_Model_A, "40 selected held-out cases"), metric_row("Version B - environmental only", cases.AC_POWER, cases.Predicted_AC_POWER_Model_B, "40 selected held-out cases")])
    comparison = pd.concat([comparison, case_metrics], ignore_index=True)

    iso_features = ["AC_POWER", "DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_DC_RATIO"]
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(train_daylight[iso_features])
    test_daylight = test_daylight.copy()
    test_daylight["ANOMALY_SCORE"] = iso.decision_function(test_daylight[iso_features])
    test_daylight["ANOMALY"] = iso.predict(test_daylight[iso_features]) == -1
    flagged_by_inverter = test_daylight.groupby(["PLANT", "PLANT_ID", "SOURCE_KEY"], as_index=False).agg(test_daylight_records=("SOURCE_KEY", "size"), flagged_records=("ANOMALY", "sum"))
    flagged_by_inverter["flagged_proportion"] = flagged_by_inverter.flagged_records / flagged_by_inverter.test_daylight_records

    cases_out = cases[["Test_Case", "PLANT_ID", "SOURCE_KEY", "DATE_TIME", "AC_POWER", "Predicted_AC_POWER", "Predicted_AC_POWER_Model_A", "Predicted_AC_POWER_Model_B", "Absolute_Error", "Percentage_Error"]].rename(columns={"PLANT_ID": "Plant_ID", "SOURCE_KEY": "Inverter_ID", "DATE_TIME": "Timestamp", "AC_POWER": "Actual_AC_POWER"})
    cases_out.to_csv(OUTPUT / "40_case_predictions.csv", index=False)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    flagged_by_inverter.to_csv(OUTPUT / "isolation_forest_test_summary.csv", index=False)
    heldout_out = test_daylight[["PLANT", "PLANT_ID", "SOURCE_KEY", "DATE_TIME", "AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"]].copy()
    heldout_out["Predicted_AC_POWER_Model_A"] = pred_a.loc[heldout_out.index]
    heldout_out["Predicted_AC_POWER_Model_B"] = pred_b.loc[heldout_out.index]
    heldout_out["Error_Model_A"] = heldout_out.AC_POWER - heldout_out.Predicted_AC_POWER_Model_A
    heldout_out["Error_Model_B"] = heldout_out.AC_POWER - heldout_out.Predicted_AC_POWER_Model_B
    heldout_out.to_csv(OUTPUT / "heldout_daylight_predictions.csv", index=False)

    plot_test = test_daylight.copy()
    plot_test["Actual_AC_POWER"] = plot_test.AC_POWER
    plot_test["Version_A_Predicted"] = pred_a.loc[plot_test.index]
    plot_test["Version_B_Predicted"] = pred_b.loc[plot_test.index]
    plot_test = plot_test.sort_values("DATE_TIME").iloc[::max(1, len(plot_test) // 5000)]
    fig = go.Figure()
    fig.add_scatter(x=plot_test.DATE_TIME, y=plot_test.Actual_AC_POWER, name="Actual AC power")
    fig.add_scatter(x=plot_test.DATE_TIME, y=plot_test.Version_A_Predicted, name="Version A predicted")
    fig.add_scatter(x=plot_test.DATE_TIME, y=plot_test.Version_B_Predicted, name="Version B predicted")
    fig.update_layout(title="Held-out daylight actual versus predicted AC power", xaxis_title="Timestamp", yaxis_title="AC_POWER")
    fig.write_html(OUTPUT / "actual_vs_predicted.html", include_plotlyjs="cdn")
    errors = pd.DataFrame({"Version A error": test_daylight.AC_POWER - pred_a.loc[test_daylight.index], "Version B error": test_daylight.AC_POWER - pred_b.loc[test_daylight.index]})
    error_fig = go.Figure([go.Histogram(x=errors["Version A error"], name="Version A"), go.Histogram(x=errors["Version B error"], name="Version B")])
    error_fig.update_layout(barmode="overlay", title="Held-out daylight error distribution", xaxis_title="Actual - predicted AC power")
    error_fig.write_html(OUTPUT / "error_distribution.html", include_plotlyjs="cdn")

    full_a = comparison[(comparison.model == "Version A - with DC_POWER") & (comparison.scope.str.contains("complete"))].iloc[0]
    full_b = comparison[(comparison.model == "Version B - environmental only") & (comparison.scope.str.contains("complete"))].iloc[0]
    flagged = int(test_daylight.ANOMALY.sum())
    report = f"""SolarGuard AI model validation report
====================================

Data and split
--------------
Merged records after invalid-record removal: {len(data):,}
Chronological cutoff: {cutoff}
Training records: {len(train):,} ({train.DATE_TIME.min()} to {train.DATE_TIME.max()})
Held-out records: {len(test):,} ({test.DATE_TIME.min()} to {test.DATE_TIME.max()})
Held-out meaningful daylight records: {len(test_daylight):,}
40-case sample: {len(cases_out)} records, selected from the held-out daylight period across inverter IDs and irradiation bands.

Random Forest comparison on complete held-out daylight test set
---------------------------------------------------------------
Version A includes DC_POWER: MAE={full_a.MAE:.4f}, RMSE={full_a.RMSE:.4f}, R2={full_a.R2:.4f}, MAPE(non-zero)={full_a.MAPE_nonzero_pct:.2f}%
Version B excludes DC_POWER: MAE={full_b.MAE:.4f}, RMSE={full_b.RMSE:.4f}, R2={full_b.R2:.4f}, MAPE(non-zero)={full_b.MAPE_nonzero_pct:.2f}%

Interpretation
--------------
Version A is useful for estimating expected AC output when contemporaneous DC_POWER is a legitimate available operating measurement. It is not a clean environmental-only counterfactual: DC_POWER is closely related to AC_POWER and can act as a prediction shortcut. Version B is the more conservative choice for judging environmental generalisation and for monitoring where DC_POWER may itself be part of the suspected fault.
The existing dashboard R2 of 0.997 should not be presented without qualification: it was calculated on the later 20 percent but across nighttime and daylight records, and it uses DC_POWER. The daylight-only holdout metrics above are the more relevant figures for this task.

Isolation Forest on held-out daylight records
---------------------------------------------
Training used only earlier daylight records; scoring used only later daylight records.
Flagged observations: {flagged:,} of {len(test_daylight):,} ({flagged / max(1, len(test_daylight)) * 100:.2f}%).
The test summary is broken down by plant and inverter in isolation_forest_test_summary.csv.
No fault labels exist, so these flags are not classification accuracy and do not prove component failure. They identify unusual patterns relative to the unsupervised training distribution and should be reviewed with irradiation, temperatures, peer performance, and physical inspection.

40-case metrics are in model_comparison.csv and row-level predictions are in 40_case_predictions.csv.
"""
    (OUTPUT / "model_evaluation_report.txt").write_text(report, encoding="utf-8")
    return {"data": data, "train": train, "test": test, "test_daylight": test_daylight, "pred_a": pred_a, "pred_b": pred_b, "comparison": comparison, "cases": cases_out, "flagged_by_inverter": flagged_by_inverter, "flagged": flagged}


if __name__ == "__main__":
    result = evaluate()
    print(result["comparison"].to_string(index=False))
    print(f"Isolation Forest flagged {result['flagged']} of {len(result['test_daylight'])} held-out daylight records.")

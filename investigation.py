"""Controlled Isolation Forest investigation on the existing held-out cohort."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from validation import load_validation_data

ROOT = Path(__file__).resolve().parent
ISO_FEATURES = ["AC_POWER", "DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_DC_RATIO"]
CONTAMINATIONS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, "auto"]
KEYS = ["PLANT", "PLANT_ID", "SOURCE_KEY", "DATE_TIME"]


def fit_score(train, test, contamination, scaled=False, features=None):
    features = features or ISO_FEATURES
    estimator = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model = make_pipeline(StandardScaler(), estimator) if scaled else estimator
    model.fit(train[features])
    score = model.decision_function(test[features])
    flagged = model.predict(test[features]) == -1
    return score, flagged


def rate_rows(test, score, flagged, setting, feature_set="current", scaled=False):
    frame = test.copy()
    frame["ANOMALY_SCORE"] = score
    frame["ANOMALY"] = flagged
    plant_rates = frame.groupby("PLANT").ANOMALY.mean().to_dict()
    return {
        "contamination": str(setting), "feature_set": feature_set, "scaled": scaled,
        "records": len(frame), "flagged_records": int(flagged.sum()),
        "flagged_percentage": float(flagged.mean() * 100),
        "affected_inverters": int(frame.loc[flagged, "SOURCE_KEY"].nunique()),
        "plant_1_percentage": float(plant_rates.get("Plant 1", 0) * 100),
        "plant_2_percentage": float(plant_rates.get("Plant 2", 0) * 100),
        "score_min": float(np.min(score)), "score_p01": float(np.quantile(score, .01)),
        "score_median": float(np.median(score)), "score_p99": float(np.quantile(score, .99)), "score_max": float(np.max(score)),
    }


def add_bins(frame):
    frame = frame.copy()
    frame["hour"] = frame.DATE_TIME.dt.hour
    frame["date"] = frame.DATE_TIME.dt.date.astype(str)
    frame["irradiation_range"] = pd.cut(frame.IRRADIATION, [-np.inf, .05, .20, .50, .80, np.inf], labels=["0.02-0.05", "0.05-0.20", "0.20-0.50", "0.50-0.80", ">0.80"])
    frame["module_temperature_range"] = pd.cut(frame.MODULE_TEMPERATURE, [-np.inf, 25, 35, 45, 55, np.inf], labels=["<25", "25-35", "35-45", "45-55", ">55"])
    frame["ambient_temperature_range"] = pd.cut(frame.AMBIENT_TEMPERATURE, [-np.inf, 20, 25, 30, 35, np.inf], labels=["<20", "20-25", "25-30", "30-35", ">35"])
    return frame


def summary(frame, group_cols):
    return frame.groupby(group_cols, observed=False, as_index=False).agg(records=("ANOMALY", "size"), flagged_records=("ANOMALY", "sum"), anomaly_percentage=("ANOMALY", "mean"), mean_score=("ANOMALY_SCORE", "mean"), mean_irradiation=("IRRADIATION", "mean"), mean_module_temperature=("MODULE_TEMPERATURE", "mean"), mean_ambient_temperature=("AMBIENT_TEMPERATURE", "mean")).assign(anomaly_percentage=lambda d: d.anomaly_percentage * 100)


def consecutive_count(frame):
    frame = frame.sort_values(KEYS).copy()
    frame["deviation_run"] = frame.groupby(["PLANT", "SOURCE_KEY"], sort=False)["MEANINGFUL_DEVIATION"].transform(lambda s: s.astype(int).groupby((~s).cumsum()).cumsum())
    return frame


def evaluate():
    data = load_validation_data()
    cutoff = data.DATE_TIME.quantile(.8)
    train = data[data.DATE_TIME <= cutoff].copy()
    test = data[(data.DATE_TIME > cutoff) & data.DAYLIGHT].copy()
    train_daylight = train[train.DAYLIGHT]

    # Reuse the already generated held-out RF predictions, rather than fitting
    # a different expected-power model for this investigation.
    heldout_path = ROOT / "heldout_daylight_predictions.csv"
    if not heldout_path.exists():
        raise FileNotFoundError("Run validation.py before investigation.py")
    expected = pd.read_csv(heldout_path, parse_dates=["DATE_TIME"])
    expected = expected[KEYS + ["Predicted_AC_POWER_Model_A"]].rename(columns={"Predicted_AC_POWER_Model_A": "EXPECTED_AC_POWER"})
    test = test.merge(expected, on=KEYS, how="inner", validate="one_to_one")
    test["DEVIATION_PCT"] = np.where(test.EXPECTED_AC_POWER > 5, (test.EXPECTED_AC_POWER - test.AC_POWER) / test.EXPECTED_AC_POWER * 100, 0)
    test["DEVIATION_PCT"] = test.DEVIATION_PCT.clip(-500, 500)
    test["MEANINGFUL_DEVIATION"] = test.DEVIATION_PCT > 15

    rows = []
    scored_by_setting = {}
    for setting in CONTAMINATIONS:
        score, flagged = fit_score(train_daylight, test, setting)
        rows.append(rate_rows(test, score, flagged, setting))
        scored_by_setting[str(setting)] = (score, flagged)
    comparison = pd.DataFrame(rows)
    current_score, current_flagged = scored_by_setting["auto"]
    train_auto_score, train_auto_flagged = fit_score(train_daylight, train_daylight, "auto")
    investigation = test.copy()
    investigation["ANOMALY_SCORE"] = current_score
    investigation["ANOMALY"] = current_flagged
    investigation = add_bins(investigation)
    investigation = consecutive_count(investigation)

    inverter_summary = summary(investigation, ["PLANT", "PLANT_ID", "SOURCE_KEY"]).sort_values("anomaly_percentage", ascending=False)
    inverter_summary["rank"] = range(1, len(inverter_summary) + 1)
    plant_summary = summary(investigation, ["PLANT"])
    hourly_summary = summary(investigation, ["hour"])
    date_summary = summary(investigation, ["date"]).sort_values("date")
    irradiation_summary = summary(investigation, ["irradiation_range"])
    module_summary = summary(investigation, ["module_temperature_range"])
    ambient_summary = summary(investigation, ["ambient_temperature_range"])

    # Feature quality: descriptive scales and pairwise Pearson correlations.
    quality_rows = []
    corr = investigation[ISO_FEATURES].corr()
    for feature in ISO_FEATURES:
        quality_rows.append({"analysis": "feature_scale", "feature": feature, "mean": investigation[feature].mean(), "std": investigation[feature].std(), "min": investigation[feature].min(), "max": investigation[feature].max()})
    for left in ISO_FEATURES:
        for right in ISO_FEATURES:
            if left < right:
                quality_rows.append({"analysis": "pairwise_correlation", "feature": left, "other_feature": right, "correlation": corr.loc[left, right]})
    feature_analysis = pd.DataFrame(quality_rows)

    # Controlled feature-set experiment at the existing auto threshold.
    alternatives = []
    for name, features, scaled in [
        ("current_unscaled", ISO_FEATURES, False),
        ("current_standard_scaled", ISO_FEATURES, True),
        ("environment_only_unscaled", ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"], False),
        ("conversion_context_unscaled", ["IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_DC_RATIO"], False),
    ]:
        score, flagged = fit_score(train_daylight, test, "auto", scaled=scaled, features=features)
        row = rate_rows(test, score, flagged, "auto", name, scaled)
        row["features"] = ", ".join(features)
        row["overlap_with_current"] = float(np.mean(flagged == current_flagged))
        alternatives.append(row)
    alternative_table = pd.DataFrame(alternatives)

    groups = np.select([
        investigation.ANOMALY & investigation.MEANINGFUL_DEVIATION,
        investigation.ANOMALY & ~investigation.MEANINGFUL_DEVIATION,
        ~investigation.ANOMALY & investigation.MEANINGFUL_DEVIATION,
    ], ["A: anomaly flagged + meaningful power deviation", "B: anomaly flagged + no meaningful power deviation", "C: no anomaly + meaningful power deviation"], default="D: neither anomaly nor meaningful power deviation")
    investigation["deviation_group"] = groups
    group_table = investigation.groupby("deviation_group", as_index=False).agg(records=("ANOMALY", "size"), percentage=("ANOMALY", "mean"), mean_deviation_pct=("DEVIATION_PCT", "mean"), flagged_records=("ANOMALY", "sum"), inverters=("SOURCE_KEY", "nunique"))
    group_table["percentage"] = group_table.records / len(investigation) * 100

    # Export required artifacts and a few useful supporting tables.
    comparison.to_csv(ROOT / "contamination_comparison.csv", index=False)
    inverter_summary.to_csv(ROOT / "inverter_anomaly_summary.csv", index=False)
    plant_summary.to_csv(ROOT / "plant_anomaly_summary.csv", index=False)
    hourly_summary.to_csv(ROOT / "hourly_anomaly_summary.csv", index=False)
    date_summary.to_csv(ROOT / "date_anomaly_summary.csv", index=False)
    feature_analysis.to_csv(ROOT / "anomaly_feature_analysis.csv", index=False)
    group_table.to_csv(ROOT / "anomaly_deviation_groups.csv", index=False)
    alternative_table.to_csv(ROOT / "anomaly_feature_set_comparison.csv", index=False)
    investigation.to_csv(ROOT / "heldout_anomaly_investigation_records.csv", index=False)
    irradiation_summary.to_csv(ROOT / "irradiation_anomaly_summary.csv", index=False)
    module_summary.to_csv(ROOT / "module_temperature_anomaly_summary.csv", index=False)
    ambient_summary.to_csv(ROOT / "ambient_temperature_anomaly_summary.csv", index=False)

    def chart(table, x, y, title, filename, color=None):
        fig = px.bar(table, x=x, y=y, color=color, title=title, text_auto=".1f")
        fig.write_html(ROOT / filename, include_plotlyjs="cdn")

    chart(inverter_summary.head(44), "SOURCE_KEY", "anomaly_percentage", "Anomaly percentage by inverter", "anomaly_percentage_by_inverter.html", "PLANT")
    chart(plant_summary, "PLANT", "anomaly_percentage", "Anomaly percentage by plant", "anomaly_percentage_by_plant.html")
    chart(hourly_summary, "hour", "anomaly_percentage", "Hourly anomaly distribution", "hourly_anomaly_distribution.html")
    chart(irradiation_summary, "irradiation_range", "anomaly_percentage", "Anomaly distribution across irradiation ranges", "irradiation_anomaly_distribution.html")
    temp_chart = pd.concat([module_summary.assign(range_type="Module temperature", temperature_range=module_summary.module_temperature_range.astype(str).values).rename(columns={"module_temperature_range": "range"}), ambient_summary.assign(range_type="Ambient temperature", temperature_range=ambient_summary.ambient_temperature_range.astype(str).values).rename(columns={"ambient_temperature_range": "range"})], ignore_index=True)
    chart(temp_chart, "range", "anomaly_percentage", "Anomaly distribution across temperature ranges", "temperature_anomaly_distribution.html", "range_type")
    chart(comparison, "contamination", "flagged_percentage", "Contamination setting comparison", "contamination_setting_comparison.html")

    high = inverter_summary.iloc[0]
    low = inverter_summary.iloc[-1]
    top_dates = date_summary.sort_values("flagged_records", ascending=False).head(10)
    date_lines = "\n".join(f"{row.date}: {int(row.flagged_records):,}/{int(row.records):,} flagged ({row.anomaly_percentage:.2f}%)" for row in top_dates.itertuples())
    group_lines = "\n".join(f"{row.deviation_group}: {int(row.records):,} records ({row.percentage:.2f}%)" for row in group_table.itertuples())
    report = f"""SolarGuard AI Isolation Forest investigation
===============================================

Code-verified implementation
----------------------------
Current Isolation Forest features: {', '.join(ISO_FEATURES)}.
Current contamination: 'auto' in both app.py and validation.py. 'auto' lets sklearn use its default offset rather than specifying a verified fault rate.
Features are not scaled or normalized. AC_POWER and DC_POWER are on much larger numeric scales than temperatures and irradiation; trees can still split all features, but scale diagnostics and correlated features matter for interpretation.
Validation uses strict IRRADIATION > 0.02, not AC_POWER as a daylight proxy. The held-out model is trained only on earlier daylight observations; test scoring uses later daylight observations. The app.py production model differs: it fits Isolation Forest on all available daylight records and uses `(IRRADIATION > 0.02) OR (AC_POWER > 0.1)`.
Labels are produced by `predict`: -1 becomes anomaly and +1 becomes non-anomaly. `decision_function` is retained as the continuous score; lower scores are more abnormal.

Held-out cohort
---------------
Total strict held-out daylight records: {len(investigation):,}
Flagged with current 'auto' setting: {int(current_flagged.sum()):,}
Unflagged: {int((~current_flagged).sum()):,}
Overall anomaly percentage: {current_flagged.mean() * 100:.2f}%
Training-baseline records scored with the same auto model: {len(train_daylight):,}; labels on that training baseline: {int(train_auto_flagged.sum()):,} ({train_auto_flagged.mean() * 100:.2f}%). This confirms that sklearn's 'auto' boundary is not behaving as a 10% or 1% fault prevalence estimate here.
Plant 1 anomaly percentage: {plant_summary.loc[plant_summary.PLANT == 'Plant 1', 'anomaly_percentage'].iloc[0]:.2f}%
Plant 2 anomaly percentage: {plant_summary.loc[plant_summary.PLANT == 'Plant 2', 'anomaly_percentage'].iloc[0]:.2f}%
Highest inverter rate: {high.SOURCE_KEY} ({high.PLANT}), {high.anomaly_percentage:.2f}%
Lowest inverter rate: {low.SOURCE_KEY} ({low.PLANT}), {low.anomaly_percentage:.2f}%
Affected inverters: {investigation.loc[investigation.ANOMALY, 'SOURCE_KEY'].nunique()} of {investigation.SOURCE_KEY.nunique()}

Temporal and environmental interpretation
------------------------------------------
Top dates by flagged count:
{date_lines}
Inverter and plant tables contain rates, counts, mean scores, and environmental means. These are statistical pattern summaries, not fault evidence. High rates spread across many inverters and both plants suggest the current unsupervised boundary is describing distributional variation, operating-regime transitions, feature redundancy, or temporal drift rather than isolating a small set of confirmed failing inverters.

Four-way anomaly/deviation comparison
--------------------------------------
Meaningful deviation means expected AC power > 5 and `(expected - actual) / expected > 15%`. This is a provisional screening rule, not a validated fault limit.
{group_lines}
A means a statistical anomaly and output shortfall coincide and is the highest-priority review group. B is unusual without meaningful shortfall. C is output shortfall without an Isolation Forest flag. D has neither signal. None is a confirmed fault category.

Contamination experiments
-------------------------
contamination_comparison.csv reports all requested settings, rates, affected inverters, plant rates, and score quantiles. Contamination is a threshold-selection assumption, not a measured fault prevalence. The current 'auto' setting is not defensible as a fault rate solely because it produced {current_flagged.mean() * 100:.2f}% flags.

Feature quality and controlled alternatives
-------------------------------------------
anomaly_feature_analysis.csv contains feature scales and pairwise correlations. anomaly_feature_set_comparison.csv compares the current unscaled set, standard-scaled set, environment-only set, and conversion-context set using the same train/test rows and auto threshold. Scaling or feature removal should be judged by stability, interpretability, peer consistency, and agreement with physical evidence; a lower flag rate alone is not success.

Recommendation
--------------
Present Isolation Forest as an investigation-prioritisation signal, not a fault detector. Report the 37.63% original validation rate and explain that the strict investigation cohort and current 'auto' threshold produce the rate shown in contamination_comparison.csv. Show plant/inverter concentration, environmental distributions, and the four groups before proposing a threshold. For production, fit only on a clean historical baseline, validate a contamination policy with operator-reviewed events, consider per-plant or per-regime baselines, and combine anomaly flags with repeated output deviations. No precision, recall, F1, or fault accuracy can be claimed without labels.
"""
    (ROOT / "isolation_forest_investigation.txt").write_text(report, encoding="utf-8")
    return {"investigation": investigation, "comparison": comparison, "inverter_summary": inverter_summary, "plant_summary": plant_summary, "hourly_summary": hourly_summary, "feature_analysis": feature_analysis, "groups": group_table, "alternatives": alternative_table}


if __name__ == "__main__":
    result = evaluate()
    print(result["comparison"].to_string(index=False))
    print(result["groups"].to_string(index=False))

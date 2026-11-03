from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SolarGuard AI", page_icon="☀", layout="wide")
st.markdown("""<style>
.stApp { background: #fbf4e8; color: #3d2b22; }
[data-testid="stSidebar"] { background: #eee3d2; }
h1,h2,h3 { color: #754735; }
div[data-testid="stMetric"] { background: #f5ead9; border-left: 4px solid #c56745; padding: 10px; }
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
FILES = {
    "Plant 1": ("Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv"),
    "Plant 2": ("Plant_2_Generation_Data.csv", "Plant_2_Weather_Sensor_Data.csv"),
}
REQ_GEN = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"}
REQ_WEATHER = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"}

@st.cache_data(show_spinner="Loading and merging plant data...")
def load_data():
    frames = []
    schema = {}
    for plant, (gen_name, weather_name) in FILES.items():
        gen_paths = list(ROOT.rglob(gen_name))
        weather_paths = list(ROOT.rglob(weather_name))
        if not gen_paths or not weather_paths:
            raise FileNotFoundError(f"Missing {gen_name} or {weather_name}")
        generation = pd.read_csv(gen_paths[0])
        weather = pd.read_csv(weather_paths[0])
        schema[plant] = {"generation": list(generation.columns), "weather": list(weather.columns)}
        if not REQ_GEN.issubset(generation.columns) or not REQ_WEATHER.issubset(weather.columns):
            raise ValueError(f"Unexpected columns in {plant} files")
        generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], format="mixed", dayfirst=True, errors="coerce")
        weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format="mixed", errors="coerce")
        generation["PLANT"] = plant
        weather = weather.dropna(subset=["DATE_TIME", "PLANT_ID"])
        # Weather is a plant-level sensor: deduplicate only on its plant/time key.
        weather = weather.groupby(["PLANT_ID", "DATE_TIME"], as_index=False)[
            ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean()
        merged = generation.merge(weather, on=["PLANT_ID", "DATE_TIME"], how="left", validate="many_to_one")
        frames.append(merged)
    data = pd.concat(frames, ignore_index=True)
    numeric = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["DATE_TIME"] = pd.to_datetime(data["DATE_TIME"], errors="coerce")
    data = data.dropna(subset=["DATE_TIME", "SOURCE_KEY", "AC_POWER", "DC_POWER"]).sort_values("DATE_TIME").reset_index(drop=True)
    data[numeric] = data[numeric].fillna(0)
    data["HOUR"] = data.DATE_TIME.dt.hour + data.DATE_TIME.dt.minute / 60
    data["DAY"] = data.DATE_TIME.dt.day_of_year
    data["HOUR_SIN"] = np.sin(2 * np.pi * data.HOUR / 24)
    data["HOUR_COS"] = np.cos(2 * np.pi * data.HOUR / 24)
    data["AC_DC_RATIO"] = np.where(data.DC_POWER > 0, data.AC_POWER / data.DC_POWER, 0).clip(0, 2)
    # Match the validated cohort: use measured plant irradiation rather than
    # AC power as a daylight proxy, which can admit transition records.
    data["DAYLIGHT"] = data.IRRADIATION > 0.02
    return data, schema

@st.cache_resource(show_spinner="Training models...")
def train_models(data):
    features = ["DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "HOUR", "DAY", "HOUR_SIN", "HOUR_COS", "SOURCE_KEY"]
    cutoff = data.DATE_TIME.quantile(0.8)
    train = data[data.DATE_TIME <= cutoff]
    test = data[data.DATE_TIME > cutoff]
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Not enough chronological records for a train/test split")
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["SOURCE_KEY"])], remainder="passthrough")
    rf = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1, min_samples_leaf=2))])
    rf.fit(train[features], train.AC_POWER)
    data = data.copy()
    data["EXPECTED_AC_POWER"] = rf.predict(data[features])
    test_pred = data.loc[test.index, "EXPECTED_AC_POWER"]
    metrics = {"MAE": mean_absolute_error(test.AC_POWER, test_pred), "RMSE": mean_squared_error(test.AC_POWER, test_pred) ** 0.5, "R2": r2_score(test.AC_POWER, test_pred)}
    iso_features = ["AC_POWER", "DC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_DC_RATIO"]
    daylight = data.DAYLIGHT
    # Fit the unsupervised baseline only on observations before the same
    # chronological cutoff used by the validated Random Forest split.
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso.fit(train.loc[train.DAYLIGHT, iso_features])
    data["ANOMALY_SCORE"] = 0.0
    data.loc[daylight, "ANOMALY_SCORE"] = iso.decision_function(data.loc[daylight, iso_features])
    data["ANOMALY"] = False
    data.loc[daylight, "ANOMALY"] = iso.predict(data.loc[daylight, iso_features]) == -1
    data["DEVIATION_PCT"] = np.where(data.EXPECTED_AC_POWER > 5, (data.EXPECTED_AC_POWER - data.AC_POWER) / data.EXPECTED_AC_POWER * 100, 0)
    data["DEVIATION_PCT"] = data.DEVIATION_PCT.clip(-500, 500)
    data["MEANINGFUL_DEVIATION"] = (data.EXPECTED_AC_POWER > 5) & (data.DEVIATION_PCT > 15)
    data["REPEATED_DEVIATIONS"] = data.groupby(["PLANT", "SOURCE_KEY"], sort=False)["MEANINGFUL_DEVIATION"].transform(lambda values: values.rolling(3, min_periods=1).sum() >= 2)
    data["ANOMALY_REASON"] = "No statistical anomaly flagged."
    anomaly_rows = data.ANOMALY & data.DAYLIGHT
    data.loc[anomaly_rows, "ANOMALY_REASON"] = data.loc[anomaly_rows].apply(lambda row: f"Unusual pattern at irradiation {row.IRRADIATION:.3f}, module {row.MODULE_TEMPERATURE:.1f}C, ambient {row.AMBIENT_TEMPERATURE:.1f}C; actual AC {row.AC_POWER:.1f} vs expected {row.EXPECTED_AC_POWER:.1f}.", axis=1)
    low_irradiation = anomaly_rows & (data.IRRADIATION <= 0.05)
    data.loc[low_irradiation, "ANOMALY_REASON"] = data.loc[low_irradiation].apply(lambda row: f"Low irradiation transition ({row.IRRADIATION:.3f}); actual AC {row.AC_POWER:.1f} vs expected {row.EXPECTED_AC_POWER:.1f}. Treat as regime variation, not a confirmed fault.", axis=1)
    meaningful = anomaly_rows & data.MEANINGFUL_DEVIATION
    data.loc[meaningful, "ANOMALY_REASON"] = data.loc[meaningful].apply(lambda row: f"Actual AC {row.AC_POWER:.1f} is {row.DEVIATION_PCT:.1f}% below expected {row.EXPECTED_AC_POWER:.1f} at irradiation {row.IRRADIATION:.3f}; module {row.MODULE_TEMPERATURE:.1f}C.", axis=1)
    repeated = meaningful & data.REPEATED_DEVIATIONS
    data.loc[repeated, "ANOMALY_REASON"] = data.loc[repeated].apply(lambda row: f"Repeated deviation: actual AC {row.AC_POWER:.1f} is {row.DEVIATION_PCT:.1f}% below expected {row.EXPECTED_AC_POWER:.1f} across recent readings; irradiation {row.IRRADIATION:.3f}.", axis=1)
    hot = anomaly_rows & (data.MODULE_TEMPERATURE >= 55)
    data.loc[hot & ~repeated, "ANOMALY_REASON"] = data.loc[hot & ~repeated].apply(lambda row: f"Unusual pattern with module temperature {row.MODULE_TEMPERATURE:.1f}C; actual AC {row.AC_POWER:.1f} vs expected {row.EXPECTED_AC_POWER:.1f}.", axis=1)
    return data, rf, iso, metrics, features, iso_features

def make_report(data):
    grouped = data.groupby(["PLANT", "SOURCE_KEY"], as_index=False).agg(
        records=("SOURCE_KEY", "size"), actual_ac_power=("AC_POWER", "sum"), expected_ac_power=("EXPECTED_AC_POWER", "sum"),
        mean_deviation_pct=("DEVIATION_PCT", "mean"), anomalous_records=("ANOMALY", "sum"), daylight_records=("DAYLIGHT", "sum"))
    grouped["deviation_pct"] = np.where(grouped.expected_ac_power > 5, (grouped.expected_ac_power - grouped.actual_ac_power) / grouped.expected_ac_power * 100, 0)
    grouped["anomaly_rate"] = grouped.anomalous_records / grouped.daylight_records.clip(lower=1)
    grouped["status"] = "NORMAL"
    concern = ((grouped.deviation_pct > 15) & (grouped.daylight_records >= 10)) | ((grouped.deviation_pct > 8) & (grouped.anomaly_rate > 0.10))
    grouped.loc[concern, "status"] = "POTENTIALLY UNDERPERFORMING"
    inspect = (grouped.deviation_pct > 25) & (grouped.anomaly_rate > 0.10) & (grouped.daylight_records >= 10)
    grouped.loc[inspect, "status"] = "NEEDS INSPECTION"
    grouped["recommendation"] = np.select([grouped.status.eq("NEEDS INSPECTION"), grouped.status.eq("POTENTIALLY UNDERPERFORMING")], ["Prioritize physical inspection after reviewing sensor evidence.", "Review repeated deviations and compare peers under similar irradiation."], default="Continue routine monitoring.")
    return grouped

def chart_layout(fig):
    fig.update_layout(template="simple_white", paper_bgcolor="#fbf4e8", plot_bgcolor="#fffaf2", font_color="#3d2b22", margin=dict(l=20, r=20, t=45, b=20))
    return fig

try:
    raw, schema = load_data()
    scored, rf, iso, metrics, model_features, iso_features = train_models(raw)
    report = make_report(scored)
except Exception as exc:
    st.title("SolarGuard AI")
    st.error(f"The data pipeline could not start: {exc}")
    st.info("Place the four Kaggle CSV files anywhere below this project directory, then reload the app.")
    st.stop()

st.sidebar.title("SolarGuard AI")
st.sidebar.caption("Evidence-led inverter monitoring")
page = st.sidebar.radio("Navigate", ["Overview", "Inverter Monitoring", "Anomaly Detection", "Model Performance", "Model Validation", "Dataset Explorer"])
plants = ["All plants"] + sorted(scored.PLANT.unique().tolist())
plant_filter = st.sidebar.selectbox("Plant", plants)
view = scored if plant_filter == "All plants" else scored[scored.PLANT == plant_filter]
st.sidebar.caption(f"Coverage: {scored.DATE_TIME.min():%d %b %Y} to {scored.DATE_TIME.max():%d %b %Y}")

st.title("SolarGuard AI")
st.caption("Intelligent solar inverter performance monitoring and anomaly detection")

if page == "Overview":
    st.subheader("Fleet overview")
    cols = st.columns(7)
    vals = [view.PLANT.nunique(), view.SOURCE_KEY.nunique(), len(view), int((report[report.PLANT.isin(view.PLANT.unique())].status == "NORMAL").sum()), int((report[report.PLANT.isin(view.PLANT.unique())].status == "POTENTIALLY UNDERPERFORMING").sum()), int((report[report.PLANT.isin(view.PLANT.unique())].status == "NEEDS INSPECTION").sum()), view.AC_POWER.sum()]
    labels = ["Plants", "Inverters", "Records", "Normal", "Potential concern", "Needs inspection", "Actual AC power"]
    for col, label, val in zip(cols, labels, vals): col.metric(label, f"{val:,.0f}" if isinstance(val, (int, float, np.number)) else val)
    st.info(f"Selected scope: {plant_filter} | {view.DATE_TIME.min():%d %b %Y %H:%M} to {view.DATE_TIME.max():%d %b %Y %H:%M}")
    plant_chart = view.groupby(["DATE_TIME", "PLANT"], as_index=False).AC_POWER.sum()
    st.plotly_chart(chart_layout(px.line(plant_chart, x="DATE_TIME", y="AC_POWER", color="PLANT", title="Plant-wise actual AC generation")), use_container_width=True)
    inv_chart = view.groupby("SOURCE_KEY", as_index=False).AC_POWER.sum().sort_values("AC_POWER", ascending=False).head(20)
    st.plotly_chart(chart_layout(px.bar(inv_chart, x="AC_POWER", y="SOURCE_KEY", orientation="h", title="Top inverter generation")), use_container_width=True)
elif page == "Inverter Monitoring":
    st.subheader("Inverter monitoring")
    plant = st.selectbox("Plant", sorted(scored.PLANT.unique()))
    pv = scored[scored.PLANT == plant]
    inverter = st.selectbox("Inverter", sorted(pv.SOURCE_KEY.unique()))
    iv = pv[pv.SOURCE_KEY == inverter]
    start, end = st.slider("Date/time range", iv.DATE_TIME.min().to_pydatetime(), iv.DATE_TIME.max().to_pydatetime(), (iv.DATE_TIME.min().to_pydatetime(), iv.DATE_TIME.max().to_pydatetime()))
    iv = iv[iv.DATE_TIME.between(start, end)]
    fig = go.Figure([go.Scatter(x=iv.DATE_TIME, y=iv.AC_POWER, name="Actual AC"), go.Scatter(x=iv.DATE_TIME, y=iv.EXPECTED_AC_POWER, name="Expected AC", line=dict(color="#c56745"))])
    st.plotly_chart(chart_layout(fig.update_layout(title="Actual vs model-estimated expected AC power")), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(chart_layout(px.line(iv, x="DATE_TIME", y="DEVIATION_PCT", title="Power deviation (%)")), use_container_width=True)
    c2.plotly_chart(chart_layout(px.line(iv, x="DATE_TIME", y="IRRADIATION", title="Plant-level irradiation")), use_container_width=True)
    env = iv.melt(id_vars="DATE_TIME", value_vars=["MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE"], var_name="sensor", value_name="temperature")
    st.plotly_chart(chart_layout(px.line(env, x="DATE_TIME", y="temperature", color="sensor", title="Temperature readings")), use_container_width=True)
    st.dataframe(iv[["DATE_TIME", "AC_POWER", "EXPECTED_AC_POWER", "DEVIATION_PCT", "ANOMALY_SCORE", "ANOMALY"]].tail(100), use_container_width=True)
elif page == "Anomaly Detection":
    st.subheader("Anomaly detection queue")
    status = st.selectbox("Status", ["All"] + sorted(report.status.unique()))
    flagged = report if status == "All" else report[report.status == status]
    st.dataframe(flagged, use_container_width=True)
    records = scored[scored.ANOMALY & scored.DAYLIGHT].merge(report[["PLANT", "SOURCE_KEY", "status", "recommendation"]], on=["PLANT", "SOURCE_KEY"])
    st.caption("Statistical anomalies are investigation signals, not confirmed inverter faults. Reasons combine irradiation, temperatures, model expectation, actual output, and recent deviations.")
    st.dataframe(records[["PLANT", "SOURCE_KEY", "DATE_TIME", "AC_POWER", "EXPECTED_AC_POWER", "DEVIATION_PCT", "ANOMALY_SCORE", "ANOMALY_REASON", "status", "recommendation"]].sort_values("DATE_TIME", ascending=False), use_container_width=True)
    st.plotly_chart(chart_layout(px.scatter(view, x="IRRADIATION", y="AC_POWER", color="ANOMALY", hover_data=["SOURCE_KEY", "DATE_TIME"], title="Irradiation vs AC power")), use_container_width=True)
elif page == "Model Performance":
    st.subheader("Model performance")
    st.caption("Random Forest uses an 80/20 chronological split. Metrics are calculated on the held-out tail of the observed dataset.")
    a, b, c = st.columns(3)
    a.metric("MAE", f"{metrics['MAE']:.2f}"); b.metric("RMSE", f"{metrics['RMSE']:.2f}"); c.metric("R²", f"{metrics['R2']:.3f}")
    test_start = scored.DATE_TIME.quantile(0.8)
    test_view = scored[scored.DATE_TIME > test_start].sample(min(5000, (scored.DATE_TIME > test_start).sum()), random_state=42).sort_values("DATE_TIME")
    st.plotly_chart(chart_layout(px.scatter(test_view, x="AC_POWER", y="EXPECTED_AC_POWER", color="PLANT", title="Held-out actual vs predicted AC power", labels={"EXPECTED_AC_POWER":"Predicted expected AC power"})), use_container_width=True)
    residual = test_view.AC_POWER - test_view.EXPECTED_AC_POWER
    st.plotly_chart(chart_layout(px.histogram(x=residual, nbins=50, title="Held-out residual distribution", labels={"x":"Actual - predicted"})), use_container_width=True)
    st.plotly_chart(chart_layout(px.histogram(view[view.DAYLIGHT].ANOMALY_SCORE, nbins=50, title="Isolation Forest anomaly score distribution")), use_container_width=True)
elif page == "Model Validation":
    st.subheader("Unseen-data model validation")
    comparison_path = ROOT / "model_comparison.csv"
    cases_path = ROOT / "40_case_predictions.csv"
    heldout_path = ROOT / "heldout_daylight_predictions.csv"
    if not comparison_path.exists() or not cases_path.exists() or not heldout_path.exists():
        st.warning("Validation artifacts are not available. Run `python validation.py` from the project directory.")
    else:
        comparison = pd.read_csv(comparison_path)
        cases = pd.read_csv(cases_path)
        heldout = pd.read_csv(heldout_path, parse_dates=["DATE_TIME"])
        st.info("These figures are from models trained on earlier observations and evaluated only on later meaningful-daylight records. They are separate from the original dashboard model view.")
        full = comparison[comparison.scope.str.contains("complete")]
        cases_metrics = comparison[comparison.scope.str.contains("40 selected")]
        st.markdown("**Complete held-out daylight test set**")
        st.dataframe(full, hide_index=True, use_container_width=True)
        st.markdown("**40 genuine held-out cases**")
        st.dataframe(cases_metrics, hide_index=True, use_container_width=True)
        st.caption("The 40-case file contains actual measured AC_POWER ground truth; Version A is the current DC_POWER-assisted model and Version B excludes DC_POWER.")
        st.dataframe(cases, hide_index=True, use_container_width=True)
        sample = heldout.sort_values("DATE_TIME").iloc[::max(1, len(heldout) // 3000)]
        fig = go.Figure([go.Scatter(x=sample.DATE_TIME, y=sample.AC_POWER, name="Actual AC power"), go.Scatter(x=sample.DATE_TIME, y=sample.Predicted_AC_POWER_Model_A, name="Version A predicted"), go.Scatter(x=sample.DATE_TIME, y=sample.Predicted_AC_POWER_Model_B, name="Version B predicted")])
        st.plotly_chart(chart_layout(fig.update_layout(title="Held-out daylight actual versus predicted AC power")), use_container_width=True)
        errors = pd.DataFrame({"Version A": heldout.Error_Model_A, "Version B": heldout.Error_Model_B}).melt(var_name="Model", value_name="Error")
        st.plotly_chart(chart_layout(px.histogram(errors, x="Error", color="Model", barmode="overlay", nbins=60, title="Held-out daylight error distribution")), use_container_width=True)
        st.download_button("Download validation report", (ROOT / "model_evaluation_report.txt").read_bytes(), "model_evaluation_report.txt", "text/plain")
        investigation_path = ROOT / "heldout_anomaly_investigation_records.csv"
        if investigation_path.exists() and (ROOT / "contamination_comparison.csv").exists():
            st.divider()
            st.subheader("Isolation Forest Investigation")
            st.caption("Statistical anomaly signals are not confirmed inverter faults. This investigation uses the strict held-out daylight cohort and earlier-data training baseline.")
            anomalies = pd.read_csv(investigation_path, parse_dates=["DATE_TIME"])
            contamination = pd.read_csv(ROOT / "contamination_comparison.csv")
            inverter_anomalies = pd.read_csv(ROOT / "inverter_anomaly_summary.csv")
            plant_anomalies = pd.read_csv(ROOT / "plant_anomaly_summary.csv")
            hourly_anomalies = pd.read_csv(ROOT / "hourly_anomaly_summary.csv")
            groups = pd.read_csv(ROOT / "anomaly_deviation_groups.csv")
            m1, m2, m3, m4 = st.columns(4)
            current_row = contamination[contamination.contamination.astype(str) == "auto"].iloc[0]
            m1.metric("Held-out daylight records", f"{int(current_row.records):,}")
            m2.metric("Current flagged", f"{int(current_row.flagged_records):,}")
            m3.metric("Current anomaly rate", f"{current_row.flagged_percentage:.2f}%")
            m4.metric("Contamination", "auto")
            p1, p2 = st.columns(2)
            p1.plotly_chart(chart_layout(px.bar(plant_anomalies, x="PLANT", y="anomaly_percentage", title="Anomaly percentage by plant", text_auto=".1f")), use_container_width=True)
            p2.plotly_chart(chart_layout(px.bar(inverter_anomalies, x="SOURCE_KEY", y="anomaly_percentage", color="PLANT", title="Anomaly percentage by inverter", text_auto=".1f")), use_container_width=True)
            st.dataframe(inverter_anomalies[["rank", "PLANT", "SOURCE_KEY", "records", "flagged_records", "anomaly_percentage", "mean_irradiation", "mean_module_temperature"]], hide_index=True, use_container_width=True)
            st.plotly_chart(chart_layout(px.line(hourly_anomalies, x="hour", y="anomaly_percentage", markers=True, title="Hourly anomaly pattern")), use_container_width=True)
            st.plotly_chart(chart_layout(px.line(contamination, x="contamination", y="flagged_percentage", markers=True, title="Controlled contamination comparison")), use_container_width=True)
            st.dataframe(contamination[["contamination", "flagged_records", "flagged_percentage", "affected_inverters", "plant_1_percentage", "plant_2_percentage", "score_p01", "score_median", "score_p99"]], hide_index=True, use_container_width=True)
            st.markdown("**Anomaly versus potential underperformance**")
            st.dataframe(groups, hide_index=True, use_container_width=True)
            st.info("Group A combines an anomaly flag with a provisional >15% expected-output shortfall. Group B is anomalous without that shortfall, Group C is a shortfall without an anomaly flag, and Group D has neither. None is a confirmed fault category.")
else:
    st.subheader("Dataset explorer")
    st.write("Verified source columns", schema)
    a, b, c, d = st.columns(4)
    a.metric("Records", len(view)); b.metric("Inverters", view.SOURCE_KEY.nunique()); c.metric("Missing values", int(view.isna().sum().sum())); d.metric("Columns", len(view.columns))
    st.write("Time coverage", view.DATE_TIME.min(), "to", view.DATE_TIME.max())
    st.dataframe(view.head(100), use_container_width=True)
    descriptions = pd.DataFrame({"Column": view.columns, "Description": ["Timestamp" if c == "DATE_TIME" else "Inverter or plant identifier" if c in ["SOURCE_KEY", "PLANT", "PLANT_ID"] else "Measured or derived numeric feature" for c in view.columns]})
    st.dataframe(descriptions, hide_index=True, use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Export reports")
for name, frame in [("inverter_performance_report.csv", report), ("anomaly_detection_report.csv", scored[scored.ANOMALY & scored.DAYLIGHT]), ("plant_summary.csv", scored.groupby("PLANT", as_index=False).agg(records=("PLANT", "size"), inverters=("SOURCE_KEY", "nunique"), actual_ac_power=("AC_POWER", "sum"), expected_ac_power=("EXPECTED_AC_POWER", "sum"), anomalies=("ANOMALY", "sum")))]:
    st.sidebar.download_button(name.replace("_", " ").title(), frame.to_csv(index=False).encode("utf-8"), name, "text/csv")
st.sidebar.caption("Thresholds are provisional prioritisation rules, not validated fault limits.")

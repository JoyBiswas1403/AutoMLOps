import os
import mlflow
import pandas as pd
import streamlit as st

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.set_page_config(page_title="Experiment Comparison", layout="wide")
st.title("Experiment Comparison (MLflow)")

exp_name = st.text_input("Experiment name", value="e2e_mlops")
client = mlflow.tracking.MlflowClient()

exps = {e.name: e.experiment_id for e in client.list_experiments()}
if exp_name not in exps:
    st.warning("Experiment not found")
    st.stop()

runs = client.search_runs([exps[exp_name]], order_by=["attributes.start_time DESC"], max_results=50)

if not runs:
    st.info("No runs found")
    st.stop()

cols = ["run_id", "start_time", "end_time", "status", "params.hidden_units", "metrics.test_auc", "metrics.test_acc"]
rows = []
for r in runs:
    rows.append([
        r.info.run_id,
        r.info.start_time,
        r.info.end_time,
        r.info.status,
        r.data.params.get("hidden_units"),
        r.data.metrics.get("test_auc"),
        r.data.metrics.get("test_acc"),
    ])

df = pd.DataFrame(rows, columns=cols)
st.dataframe(df)

ids = st.multiselect("Select runs to compare", options=df.run_id.tolist(), max_selections=2)
if len(ids) == 2:
    r1 = client.get_run(ids[0])
    r2 = client.get_run(ids[1])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Run A")
        st.json({"metrics": r1.data.metrics, "params": r1.data.params})
    with c2:
        st.subheader("Run B")
        st.json({"metrics": r2.data.metrics, "params": r2.data.params})

from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.python import PythonSensor
from airflow.models import Variable

# Ensure project path is available
import sys, os
sys.path.append('/app')

from training.src.train import train_and_deploy
from drift.detect_drift import detect_and_optionally_retrain
from pipelines.auto_promote import main as auto_promote_main, q as prom_q
from pipelines.promote_canary import promote_canary_to_production
from pipelines.notify import send as notify
from pipelines.traffic import set_canary
from pipelines.auto_rollback import main as auto_rollback_main

def _notify_failure(context):
    try:
        ti = context.get("ti")
        task_id = ti.task_id if ti else "unknown"
        dag_id = context.get("dag_run").dag_id if context.get("dag_run") else "unknown"
        exc = str(context.get("exception"))
        notify("Task failed", f"{dag_id}.{task_id} failed", {"exception": exc})
    except Exception:
        pass

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _notify_failure,
}

def _train():
    train_and_deploy()

def _drift():
    detect_and_optionally_retrain(simulate=False)


def _evaluate_canary() -> bool:
    # Dry run => returns True if criteria pass
    ok = bool(auto_promote_main(dry_run=True))
    if ok:
        notify("Canary passes evaluation", "Awaiting approval to promote", {})
    else:
        notify("Canary failed evaluation", "Promotion will be skipped", {})
    return ok


def _is_stable() -> bool:
    # 15m stability window: sufficient canary traffic and low error rate
    rps_canary = prom_q('sum(rate(fastapi_requests_total{route="canary",status="200"}[15m]))') or 0.0
    errs_canary = prom_q('sum(rate(fastapi_requests_total{route="canary",status!="200"}[15m]))') or 0.0
    err_rate_canary = errs_canary / max(rps_canary + errs_canary, 1e-9)
    return bool(rps_canary >= 0.02 and err_rate_canary <= 0.02)


def _approval_gate() -> bool:
    return Variable.get("PROMOTE_APPROVED", default_var="false").lower() == "true"


def _promote():
    promote_canary_to_production()
    notify("Promotion succeeded", "Canary promoted to production", {})


def _auto_rollback():
    auto_rollback_main(apply=True)


def _rollout(p: int) -> bool:
    return set_canary(p)

with DAG(
    dag_id="mlops_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 * * * *",  # hourly
    catchup=False,
    tags=["mlops"],
) as dag:
    train = PythonOperator(task_id="train_model", python_callable=_train)
    drift = PythonOperator(task_id="check_drift", python_callable=_drift)
    evaluate = ShortCircuitOperator(task_id="evaluate_canary", python_callable=_evaluate_canary)
    wait_stable = PythonSensor(task_id="wait_stability", python_callable=_is_stable, poke_interval=60, timeout=60*30, mode="poke")

    rollout10 = ShortCircuitOperator(task_id="rollout_10", python_callable=lambda: _rollout(10))
    wait10 = PythonSensor(task_id="wait_10_stable", python_callable=_is_stable, poke_interval=60, timeout=60*30, mode="poke")
    rollout25 = ShortCircuitOperator(task_id="rollout_25", python_callable=lambda: _rollout(25))
    wait25 = PythonSensor(task_id="wait_25_stable", python_callable=_is_stable, poke_interval=60, timeout=60*30, mode="poke")
    rollout50 = ShortCircuitOperator(task_id="rollout_50", python_callable=lambda: _rollout(50))
    wait50 = PythonSensor(task_id="wait_50_stable", python_callable=_is_stable, poke_interval=60, timeout=60*30, mode="poke")
    rollout100 = ShortCircuitOperator(task_id="rollout_100", python_callable=lambda: _rollout(100))

    approval = ShortCircuitOperator(task_id="await_approval", python_callable=_approval_gate)
    promote = PythonOperator(task_id="promote_canary", python_callable=_promote)
    rollback = PythonOperator(task_id="auto_rollback", python_callable=_auto_rollback)

    train >> drift >> evaluate >> wait_stable >> rollout10 >> wait10 >> rollout25 >> wait25 >> rollout50 >> wait50 >> rollout100 >> approval >> promote >> rollback

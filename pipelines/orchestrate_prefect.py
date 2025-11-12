from __future__ import annotations
from prefect import flow, task
from training.src.train import train_and_deploy
from drift.detect_drift import detect_and_optionally_retrain


@task
def train_task():
    train_and_deploy()


@task
def monitor_task(simulate: bool = False):
    detect_and_optionally_retrain(simulate=simulate)


@flow(name="e2e_pipeline")
def orchestrate(simulate_drift: bool = False):
    train_task()
    monitor_task(simulate=simulate_drift)


if __name__ == "__main__":
    orchestrate(simulate_drift=True)

import os
from datetime import timedelta

from docker.types import Mount

VAL_SIZE = 0.25
METRICS_DIR_NAME = "/data/metrics/{{ ds }}"
GENERATE_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
TRANSFORMER_DIR_NAME = "/data/transformer_model/{{ ds }}"
MODEL_DIR_NAME = "/data/models/{{ ds }}"
PREDICTIONS_DIR_NAME = "/data/predictions/{{ ds }}"

MOUNT_OBJ = [Mount(
    source="/Users/admin/Documents/TechPark/2_sem/mlops/airflow_ml_dags/data",
    target="/data",
    type='bind'
    )]

default_args = {
    "owner": "alexander gamarnik",
    "email": ["crystall.werben@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def wait_file(file_name: str) -> bool:
    return os.path.exists(file_name)

from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


OUTPUT_DIR_NAME = "data/raw/{{ ds }}"

default_args = {
    "owner": "alexander gamarnik",
    "email": ["crystall.werben@gmail.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MOUNT_OBJ = [Mount(
    source="/Users/admin/Documents/TechPark/2_sem/mlops/airflow_ml_dags/data",
    target="/data",
    type='bind'
    )]

with DAG(
        "data_generator",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    generate = DockerOperator(
        image="airflow-generate-data",
        command=f"--out_path {OUTPUT_DIR_NAME}",
        task_id="docker-airflow-generate-data",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    generate
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utilities import MOUNT_OBJ, default_args


OUTPUT_DIR_NAME = "data/raw/{{ ds }}"


with DAG(
        "generator",
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

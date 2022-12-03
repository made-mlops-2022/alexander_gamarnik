from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from utilities import VAL_SIZE, METRICS_DIR_NAME, GENERATE_DIR_NAME, PROCESSED_DIR_NAME, TRANSFORMER_DIR_NAME, MODEL_DIR_NAME, MOUNT_OBJ, default_args


with DAG(
        "train",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    preprocess_data = DockerOperator(
        image="airflow-preprocess",
        command=f"--source_path {GENERATE_DIR_NAME} --out_path "
                f"{PROCESSED_DIR_NAME} --transform_path {TRANSFORMER_DIR_NAME}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--source_path {PROCESSED_DIR_NAME} --val_size {VAL_SIZE}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    train_model = DockerOperator(
        image="airflow-train",
        command=f"--source_path {PROCESSED_DIR_NAME} --out_path {MODEL_DIR_NAME}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    val_model = DockerOperator(
        image="airflow-validation",
        command=f"--model_source_path {MODEL_DIR_NAME} --data_source_path " \
                f"{PROCESSED_DIR_NAME} --metric_path {METRICS_DIR_NAME}",
        task_id="docker-airflow-valid",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    preprocess_data >> split_data >> train_model >> val_model
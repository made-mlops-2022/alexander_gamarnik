from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


from utilities import MOUNT_OBJ, TRANSFORMER_DIR_NAME, MODEL_DIR_NAME, GENERATE_DIR_NAME, PREDICTIONS_PATH, default_args


with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--source_path {GENERATE_DIR_NAME} --out_path {PREDICTIONS_PATH} "
                f"--transformer_path {TRANSFORMER_DIR_NAME} --model_path {MODEL_DIR_NAME}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    predict
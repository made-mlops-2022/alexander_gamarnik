# import json
import logging
# import os
import sys
# from pathlib import Path
import pickle

import click
import pandas as pd

from data import read_data
# from data.make_dataset import download_data_from_s3
from entities.predict_pipeline_params import (
    # PredictPipelineParams,
    read_predict_pipeline_params
)
from models import (
    predict_model,
)
import mlflow

# from models.model_fit_predict import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    predict_pipeline_params = read_predict_pipeline_params(config_path)

    if predict_pipeline_params.use_mlflow:

        mlflow.set_tracking_uri(predict_pipeline_params.mlflow_uri)
        mlflow.set_experiment(predict_pipeline_params.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_artifact(config_path)
            model_path, metrics = run_predict_pipeline(predict_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
    else:
        return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_pipeline_params):
    # downloading_params = predict_pipeline_params.downloading_params
    # if downloading_params:
    #     os.makedirs(downloading_params.output_folder, exist_ok=True)
    #     for path in downloading_params.paths:
    #         download_data_from_s3(
    #             downloading_params.s3_bucket,
    #             path,
    #             os.path.join(downloading_params.output_folder, Path(path).name),
    #         )

    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    test_df = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {test_df.shape}")

    with open(predict_pipeline_params.working_model_path, "rb") as file_obj:
        model = pickle.load(file_obj)
        predicts = predict_model(
            model,
            test_df,
            predict_pipeline_params.feature_params.use_log_trick
        )
        predicts_df = pd.DataFrame({predict_pipeline_params.feature_params.target_col: predicts})
        predicts_df.to_csv(predict_pipeline_params.output_data_path)
        file_obj.close()

    return


@click.command(name="train_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()

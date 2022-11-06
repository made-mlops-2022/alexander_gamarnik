import json
import logging
import os
import hydra

from data import read_data, split_train_val_data
from features.build_features import extract_target
from models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model
)


LOG_FILEPATH = "logs/training.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

os.makedirs(os.path.dirname(LOG_FILEPATH), exist_ok=True)
fh = logging.FileHandler(LOG_FILEPATH)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def train_pipeline(config):
    if config.use_mlflow:
        pass
    else:
        return run_train_pipeline(config)


def run_train_pipeline(config):
    # downloading_params = config.downloading_params
    # if downloading_params:
    #     os.makedirs(downloading_params.output_folder, exist_ok=True)
    #     for path in downloading_params.paths:
    #         download_data_from_s3(
    #             downloading_params.s3_bucket,
    #             path,
    #             os.path.join(downloading_params.output_folder, Path(path).name),
    #         )

    logger.info(f"start train pipeline with params {config}")
    data = read_data(config.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, config.splitting_params
    )

    val_target = extract_target(val_df, config.feature_params)
    train_target = extract_target(train_df, config.feature_params)
    train_df = train_df.drop(config.feature_params.target_col, axis=1)
    val_df = val_df.drop(config.feature_params.target_col, axis=1)

    val_target.to_csv(config.test_y_path, index=False)
    train_target.to_csv(config.train_y_path, index=False)
    train_df.to_csv(config.train_x_path, index=False)
    val_df.to_csv(config.test_x_path, index=False)
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    model = train_model(
        train_df, train_target, config.train_params
    )
    predicts = predict_model(
        model,
        val_df,
        config.feature_params.use_log_trick,
    )
    metrics = evaluate_model(
        predicts,
        val_target,
        config.feature_params.use_log_trick,
    )
    with open(config.metric_path + "metrics_" + config.train_params.model_type + ".json", "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        model, config.output_model_path + "model_" + config.train_params.model_type + ".pkl"
    )
    return path_to_model, metrics


if __name__ == "__main__":
    train_pipeline()

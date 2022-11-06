import logging
import os
import pickle
import pandas as pd
import hydra

from data import read_data
from models import (
    predict_model,
)

LOG_FILEPATH = "logs/predict.log"

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


@hydra.main(version_base=None, config_path="../configs", config_name="predict_config")
def predict_pipeline(config):
    if config.use_mlflow:
        pass
    else:
        return run_predict_pipeline(config)


def run_predict_pipeline(config):
    logger.info(f"start predict pipeline with params {config}")
    test_df = read_data(config.input_data_path)
    logger.info(f"data.shape is {test_df.shape}")

    with open(config.model_path + "model_" + config.model_type + ".pkl", "rb") as file_obj:
        model = pickle.load(file_obj)
        predicts = predict_model(
            model,
            test_df,
            config.feature_params.use_log_trick
        )
        predicts_df = pd.DataFrame({config.feature_params.target_col: predicts})
        predicts_df.to_csv(config.output_data_path + "prediction_" + config.model_type + ".csv")
        file_obj.close()

    return


if __name__ == "__main__":
    predict_pipeline()

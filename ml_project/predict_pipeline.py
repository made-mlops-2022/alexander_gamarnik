import logging
import pickle
import pandas as pd
import hydra

from data import read_data
from models import (
    predict_model,
)

logger = logging.getLogger(__name__)


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
    logger.info(f"model_type is {config.model_type}")

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

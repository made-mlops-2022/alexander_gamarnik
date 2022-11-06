import pandas as pd
import os
import logging
import hydra

from pandas_profiling import ProfileReport

LOG_FILEPATH = "logs/create_eda.log"

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
def make_eda_report(config):
    logger.info(config.input_data_path)
    df = pd.read_csv(config.input_data_path)
    profile = ProfileReport(df, title="EDA Report")
    profile.to_file(
        os.path.join("reports/EDA/", "EDA.html")
    )


if __name__ == "__main__":
    logger.info("=====PROGRAM START======")
    make_eda_report()
    logger.info("=====PROGRAM END======")

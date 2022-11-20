import pandas as pd
import os
import logging
import hydra
from pandas_profiling import ProfileReport

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def make_eda_report(config):
    logger.info(config.input_data_path)
    df = pd.read_csv(config.input_data_path)
    profile = ProfileReport(df, title="EDA Report")
    profile.to_file(
        os.path.join("reports/EDA/", "EDA.html")
    )


if __name__ == "__main__":
    make_eda_report()

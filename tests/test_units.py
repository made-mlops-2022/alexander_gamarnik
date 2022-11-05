import os
import sys
import json
from nbformat import read

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pytest


from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ml_project.entities.train_params import TrainingParams

sys.path.append("../")
sys.path.append(os.path.join(os.path.abspath(".")))
sys.path.append(os.path.join(os.path.abspath("."), "ml_project/"))
import ml_project.entities
from ml_project.data.make_dataset import (
    download_data_from_s3,
    read_data,
    split_train_val_data,
)
from ml_project.features.build_features import (
    process_categorical_features,
    build_categorical_pipeline,
    process_numerical_features,
    build_numerical_pipeline,
    build_transformer,
    extract_target                                 
)


TRAIN_DATA_SIZE = 100
CONFIG_FILES = [f.split(".")[0] for f in os.listdir("configs") if "yaml" in f]


def test_read_data(path="./data/processed/heart_cleveland_upload.csv"):
    data = read_data(path)
    assert isinstance(data, pd.DataFrame)


def test_split_train_val_data(synthetic_train_data, train_config):
    train_df, val_df = split_train_val_data(
        synthetic_train_data, train_config.splitting_params
    )
    assert len(val_df) / len(synthetic_train_data) == pytest.approx(
        train_config.splitting_params.val_size, 0.01
    )


def test_process_categorical_features(synthetic_train_data):
    df = process_categorical_features(synthetic_train_data)
    assert isinstance(df, pd.DataFrame)


def test_build_categorical_pipeline():
    pipeline = build_categorical_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_process_numerical_features(synthetic_train_data):
    df = process_numerical_features(synthetic_train_data)
    assert isinstance(df, pd.DataFrame)


def test_build_numerical_pipeline():
    pipeline = build_numerical_pipeline()
    assert isinstance(pipeline, Pipeline)


def test_build_transformer(train_config):
    transformer = build_transformer(
        train_config.feature_params
    )
    assert isinstance(transformer, ColumnTransformer)


def test_extract_target(synthetic_train_data, train_config):
    train_target = extract_target(
        synthetic_train_data, train_config.feature_params
    )
    assert set(train_target.values) == {0, 1}


@pytest.fixture()
def train_config():
    ml_project.entities.train_pipeline_params.register_train_configs()
    with initialize(version_base=None, config_path="../configs"):
        train_params = compose(config_name="train_config")
    return train_params


@pytest.fixture()
def synthetic_train_data():
    synthetic_data_raw = {
        "age": [29, 77],
        "sex": [1, 0],
        "chest pain": [0, 1, 2, 3],
        "resting blood pressure": [94, 200],
        "cholesterol": [126, 564],
        "fasting blood sugar": [1, 0],
        "resting electrocardiographic results": [2, 0, 1],
        "max heart rate": [71, 202],
        "exercise induced angina": [0, 1],
        "oldpeak": [0.0, 6.2],
        "slope": [1, 0, 2],
        "number of major vessels": [1, 2, 0, 3],
        "thal": [0, 2, 1],
        "condition": [0, 1],
    }
    synthetic_data = {}
    for column, values in synthetic_data_raw.items():
        synthetic_data[column] = np.random.choice(values, TRAIN_DATA_SIZE)
    synthetic_data = pd.DataFrame(synthetic_data)
    return synthetic_data


@pytest.fixture()
def synthetic_predict_data(synthetic_train_data):
    return synthetic_train_data.iloc[:50, :-1]
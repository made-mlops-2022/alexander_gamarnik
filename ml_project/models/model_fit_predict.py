import pickle
from typing import Dict, Union
# from cv2 import norm

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, f1_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

from entities.train_params import TrainingParams

SklearnRegressionModel = Union[RandomForestRegressor, LinearRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LinearRegression":
        model = LinearRegression()
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    elif train_params.model_type == "GaussianNB":
        model = GaussianNB()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: Pipeline, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnRegressionModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

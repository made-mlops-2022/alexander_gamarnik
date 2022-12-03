import os
import pandas as pd
import numpy as np
import click
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle


FEATURES_OUT_FILENAME = "features.csv"
TARGETS_OUT_FILENAME = "targets.csv"
PROCESSED_MODEL_FILENAME = "transform.pkl"


@click.command("preprocess")
@click.option("--source_path", default="../data/raw")
@click.option("--out_path", default="../data/processed")
@click.option("--transform_path", default="../data/transformer_model")
def preprocess(source_path: str, out_path: str, transform_path: str) -> None:
    os.makedirs(source_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(transform_path, exist_ok=True)

    features = pd.read_csv(os.path.join(source_path, FEATURES_OUT_FILENAME))
    targets = pd.read_csv(os.path.join(source_path, TARGETS_OUT_FILENAME))

    transformer = Pipeline([
        ("fill_na", SimpleImputer(missing_values=np.nan, strategy='mean')),
        ("scaler", StandardScaler())
    ])

    transformer.fit(features)

    features_prep = pd.DataFrame(transformer.fit_transform(features))

    features_prep.to_csv(os.path.join(out_path, FEATURES_OUT_FILENAME), index=False)
    targets.to_csv(os.path.join(out_path, TARGETS_OUT_FILENAME), index=False)

    with open(os.path.join(transform_path, PROCESSED_MODEL_FILENAME), 'wb') as fout:
        pickle.dump(transformer, fout)


if __name__ == "__main__":
    preprocess()
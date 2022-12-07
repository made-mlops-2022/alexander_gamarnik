import os
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import click

DATASET_SIZE = 100
FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"


def generate_fake_dataset(n_rows: int) -> pd.DataFrame:
    wine_dataset = load_wine()
    real_data = wine_dataset['data']
    real_targets = wine_dataset['target']

    fake_features = []
    fake_targets = []

    for _ in range(n_rows):
        row_number = np.random.randint(0, len(real_data) - 1)
        fake_features.append(real_data[row_number])
        fake_targets.append(real_targets[row_number])

    features = pd.DataFrame(fake_features)
    targets = pd.DataFrame(fake_targets)
    return features, targets


@click.command("generate")
@click.option("--out_path", default="../data/raw")
def generate(out_path: str) -> None:
    os.makedirs(out_path, exist_ok=True)

    features, targets = generate_fake_dataset(DATASET_SIZE)

    features.to_csv(os.path.join(out_path, FEATURES_FILENAME), index=False)
    targets.to_csv(os.path.join(out_path, TARGETS_FILENAME), index=False)


if __name__ == "__main__":
    generate()

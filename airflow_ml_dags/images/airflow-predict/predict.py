import os
import pandas as pd
import pickle
import click

DATA_FILENAME = "features.csv"
PREDICT_FILENAME = "predict.csv"
TRANSFORMER_FILENAME = "transform.pkl"
MODEL_FILENAME = "model.pkl"

@click.command("predict")
@click.option("--source_path", default="../data/raw")
@click.option("--out_path", default="../data/predictions")
@click.option("--transformer_path", default="../data/transformer_model/transform.pkl")
@click.option("--model_path", default="../data/models/model.pkl")
def predict(source_path: str, out_path: str, transformer_path: str, model_path: str) -> None:
    os.makedirs(source_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    data = pd.read_csv(os.path.join(source_path, DATA_FILENAME))

    with open(os.path.join(transformer_path, TRANSFORMER_FILENAME), 'rb') as file_obj:
        transformer = pickle.load(file_obj)

    with open(os.path.join(model_path, MODEL_FILENAME), 'rb') as file_obj:
        model = pickle.load(file_obj)

    transform_data = pd.DataFrame(transformer.fit_transform(data))

    predictions = pd.DataFrame(model.predict(transform_data))

    predictions.to_csv(os.path.join(out_path, PREDICT_FILENAME), index=False)


if __name__ == '__main__':
    predict()
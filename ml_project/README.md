Технопарк, МГТУ, ML-21, Гамарник Александр
===================================

MLOps Homework 1

Installation:
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Available models: LogisticRegression, GaussianNB
Train | Prediction | Create EDA report:
~~~
# Train with LogisticRegression
python ml_project/train_pipeline.py
# Train with specific model (GaussianNB, for example)
python ml_project/train_pipeline.py "train_params.model_type=GaussianNB"

# Predict
python ml_project/predict_pipeline.py
# Predict with specific model, data set and output result path
python ml_project/predict_pipeline.py "model_type=>model_name<" "input_data_path=>path_to_data<" "output_path=>output_path<"

# Create EDA report
python ml_project/create_EDA.py
~~~

Running tests with Pytest:
~~~
python -m pytest
~~~

After training, predicting, or creating EDA log information will be saved to outputs/ dir.
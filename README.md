Технопарк, МГТУ, ML-21, Гамарник Александр
===================================

MLOps Homework 1

Installation:
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Train | Prediction | Create EDA report:
~~~
# Train
python ml_project/train_pipeline.py configs/train_config.yaml
# Predict
python ml_project/predict_pipeline.py configs/predict_config.yaml
# Create EDA report
python ml_project/create_EDA.py configs/train_config.yaml
~~~

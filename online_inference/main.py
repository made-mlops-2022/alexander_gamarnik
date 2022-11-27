import os
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi_health import health

from schema import MedicalData

app = FastAPI()

model = None


@app.on_event('startup')
def load_model():
    model_path = os.getenv('MODEL_PATH')

    with open(model_path, 'rb') as f:
        global model
        model = pickle.load(f)


@app.get('/')
def home():
    return {"key": "Hello, world!"}


@app.post('/predict')
async def predict(data: MedicalData):
    data_df = pd.DataFrame([data.dict()])
    y = model.predict(data_df)
    print(y)

    condition = 'disease' if y[0] == 1 else 'no disease'
    return {'condition': condition}


def check_ready():
    return model is not None


async def success_handler(**kwargs):
    return 'Model is ready'


async def failure_handler(**kwargs):
    return 'Model is not ready'

app.add_api_route('/health', health([check_ready],
                  success_handler=success_handler,
                  failure_handler=failure_handler))

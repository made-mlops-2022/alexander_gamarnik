import json

import pytest
from fastapi.testclient import TestClient

from main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_predict_disease_endpoint():
    request = {
        "age": 59,
        "sex": 1,
        "cp": 0,
        "trestbps": 170,
        "chol": 288,
        "fbs": 0,
        "restecg": 2,
        "thalach": 159,
        "exang": 0,
        "oldpeak": 0.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'disease'}


def test_predict_no_disease_endpoint():
    request = {
        "age": 69,
        "sex": 0,
        "cp": 0,
        "trestbps": 160,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "thalach": 131,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 0,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'no disease'}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == 'Model is ready'


def test_missing_fields():
    request = {
        "age": 69,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 0,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'


def test_categorical_fields():
    request = {
        "age": 69,
        "sex": 1,
        "cp": 0,
        "trestbps": 160,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "thalach": 131,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 131,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1, 2'


def test_numerical_fields():
    request = {
        "age": 50,
        "sex": 1,
        "cp": 0,
        "trestbps": 1000,
        "chol": 234,
        "fbs": 1,
        "restecg": 2,
        "thalach": 131,
        "exang": 0,
        "oldpeak": 0.1,
        "slope": 1,
        "ca": 1,
        "thal": 0,
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'Wrong trestbps value'

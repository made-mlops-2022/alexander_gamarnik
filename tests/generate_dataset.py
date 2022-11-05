import pandas as pd
from faker import Faker
from numpy.random import normal

# from ml_project.entities import Config


def generate_dataset(n_rows: int):
    fake = Faker()
    fake_data = {
        "sex": fake.random_elements(elements=list(range(2)), length=n_rows),
        "chest pain": fake.random_elements(elements=list(range(4)), length=n_rows),
        "fasting blood sugar": fake.random_elements(elements=list(range(2)), length=n_rows),
        "resting electrocardiographic results": fake.random_elements(elements=list(range(3)), length=n_rows),
        "exercise induced angina": fake.random_elements(elements=list(range(2)), length=n_rows),
        "slope": fake.random_elements(elements=list(range(3)), length=n_rows),
        "number of major vessels": fake.random_elements(elements=list(range(4)), length=n_rows),
        "thal": fake.random_elements(elements=list(range(3)), length=n_rows),
        "age": [normal(54.54, 9.05) for _ in range(n_rows)],
        "resting blood pressure": [normal(131.69, 17.76) for _ in range(n_rows)],
        "cholesterol": [normal(247.35, 52) for _ in range(n_rows)],
        "max heart rate": [normal(149.6, 22.94) for _ in range(n_rows)],
        "oldpeak": [normal(1.06, 1.17) for _ in range(n_rows)],
        "condition": fake.random_elements(elements=list(range(2)), length=n_rows),
    }

    fake_df = pd.DataFrame(fake_data)
    return fake_df

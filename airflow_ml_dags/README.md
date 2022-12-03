[![.github/workflows/ci.yml](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml/badge.svg)](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml)

# Технопарк, МГТУ, ML-21, Гамарник Александр

MLOps Homework 3

To start:
~~~
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
~~~
FROM python:3.10-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY main.py schema.py requests data run.sh requirements.txt ./online_inference/

WORKDIR /online_inference

RUN apt-get update && apt-get install wget -y
RUN pip3 install -r requirements.txt

ENV MODEL_PATH="LogisticRegressionCV_model.pkl"

EXPOSE 15000

CMD ["bash", "run.sh"]
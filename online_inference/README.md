[![.github/workflows/ci.yml](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml/badge.svg)](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml)

Технопарк, МГТУ, ML-21, Гамарник Александр
===================================

MLOps Homework 2

Build docker image:
~~~
docker build -t homework2:v2 .
~~~

Pull docker image:
~~~
docker pull homework2:v2
~~~

Quick run:
~~~
docker run --name online_inference -p 15000:15000 homework2:v2
~~~
Service is running on _http://127.0.0.1:15000_

Run tests:
~~~
docker exec -it online_inference bash
python3 -m pytest test_main.py
~~~

Make requests:
~~~
python3 requests/make_request.py
~~~




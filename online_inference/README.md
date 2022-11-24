[![.github/workflows/ci.yml](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml/badge.svg)](https://github.com/made-mlops-2022/alexander_gamarnik/actions/workflows/ci.yaml)

# Технопарк, МГТУ, ML-21, Гамарник Александр

MLOps Homework 2

Build docker image:

```
docker build -t homework2:v2 .
```

Pull docker image:

```
docker pull homework2:v2
```

Quick run:

```
docker run --name online_inference -p 15000:15000 homework2:v2
```

Service is running on _http://127.0.0.1:15000_

Run tests:

```
docker exec -it online_inference bash
python3 -m pytest test_main.py
```

Make requests:

```
python3 requests/make_request.py
```

### Docker optimization

1. Using python:3.10-slim-bullseye [[v1]](https://hub.docker.com/layers/alexwerben/homework2/latest/images/sha256-f93f61eaee67b6aa9cc8f2d87ae35e4322309a793f3eab57b9f1738121ccb6f0?context=repo)
Size: 693 MB

2. Using python:3.10-slim-bullseye + flag --no-cache-dir when installing dependencies [[v2]](https://hub.docker.com/layers/alexwerben/homework2/v2/images/sha256-7a55a31cee67d11ef5818b1f29f5c07cce04578460590be919cbff07948703e6?context=repo)
Size: 583 MB

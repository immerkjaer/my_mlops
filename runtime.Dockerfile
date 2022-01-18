# Base image
# FROM python:3.7-slim
# FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

# install python 
# RUN apt update && \
# apt install --no-install-recommends -y build-essential gcc && \
# apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY setup.py setup.py
# COPY src/ src/
# COPY data/ data/
# COPY models/ models/
# COPY references/ references/

# WORKDIR /app
# RUN pip install -r requirements.txt --no-cache-dir

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

FROM python:3.9-slim-bullseye as base

FROM base as builder

RUN mkdir /install

WORKDIR /install

COPY src src
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY data data
COPY data.dvc data.dvc
COPY .dvcignore .dvcignore
COPY .dvc .dvc

RUN pip install -r requirements.txt --no-cache-dir

RUN dvc config core.no_scm true
RUN dvc pull

FROM base

COPY --from=builder /usr/local /usr/local

COPY src/ /app/src/
COPY --from=builder install/data /app/data/
COPY models/ /app/models/
COPY references/ /app/references/

WORKDIR /app

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
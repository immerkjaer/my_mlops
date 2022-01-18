FROM python:3.9-slim-bullseye as base

FROM base as builder

RUN mkdir /install

WORKDIR /install

COPY setup.py setup.py
COPY src src
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

FROM base

COPY --from=builder /usr/local /usr/local

COPY src/ /app/src/
COPY data/ /app/data/ 
COPY setup.py /app/setup.py
# Do not copy data

RUN pip install /app/. --no-cache-dir

WORKDIR /app

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
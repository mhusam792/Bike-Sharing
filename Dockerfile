FROM python:3.10.16-slim

WORKDIR /app

COPY pyproject.toml /app
COPY api /app/api
COPY bike_sharing_model /app/bike_sharing_model

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 8089

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8089"]

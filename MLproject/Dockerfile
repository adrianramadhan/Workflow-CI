FROM python:3.12-slim

# 1) Install MLflow + dependencies minimal
RUN pip install --no-cache-dir mlflow scikit-learn pandas uvicorn fastapi prometheus_client

# 2) Copy in artefak model hasil CI
#    Asumsikan Anda menaruh model hasil mlflow.sklearn.log_model
#    di folder `model/` di root repo sebelum build
COPY model /app/model

WORKDIR /app

# 3) Copy inference script
COPY inference.py /app/inference.py

# 4) Expose ports: 
#    - 8080 untuk MLflow serve
#    - 8000 untuk FastAPI/prometheus exporter (jika digabung)
EXPOSE 8080
EXPOSE 8000

# 5) Entrypoint: jalankan MLflow model server
ENTRYPOINT ["mlflow", "models", "serve", \
            "--model-uri", "model", \
            "--no-conda", \
            "--host", "0.0.0.0", \
            "--port", "8080"]

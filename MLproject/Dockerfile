FROM continuumio/miniconda3

# Salin seluruh folder MLproject (berisi MLproject file, conda.yaml, modelling.py, dataset)
COPY MLproject /app/MLproject

WORKDIR /app/MLproject

# Buat environment Conda sesuai conda.yaml
RUN conda env create -f conda.yaml

# Pastikan perintah selanjutnya dijalankan di environment mlflow-env
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Jalankan MLflow Project secara otomatis saat container start
ENTRYPOINT ["mlflow", "run", ".", "--env-manager=local"]

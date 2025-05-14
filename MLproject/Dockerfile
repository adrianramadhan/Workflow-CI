FROM continuumio/miniconda3

# Copy project files (MLproject, conda.yaml, modelling.py, dan dataset)
COPY MLproject conda.yaml modelling.py mental_health_dataset_preprocessing/ /app/

WORKDIR /app

# Buat environment Conda yang sesuai
RUN conda env create -f conda.yaml

# Pastikan 'conda activate' berfungsi
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

ENTRYPOINT ["mlflow", "run", ".", "--env-manager=local"]

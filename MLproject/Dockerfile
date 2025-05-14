FROM continuumio/miniconda3

COPY MLproject /app/MLproject
WORKDIR /app/MLproject

RUN conda env create -f conda.yaml

ENV PATH /opt/conda/envs/mlproject-env/bin:$PATH

ENTRYPOINT ["mlflow", "run", ".", "--no-conda"]

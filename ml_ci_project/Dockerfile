FROM continuumio/miniconda3

COPY MLProject /app/MLProject
WORKDIR /app/MLProject

RUN conda env create -f conda.yaml

ENV PATH /opt/conda/envs/mlproject-env/bin:$PATH

ENTRYPOINT ["mlflow", "run", ".", "--no-conda"]

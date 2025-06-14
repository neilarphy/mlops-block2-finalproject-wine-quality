FROM apache/airflow:2.8.3

USER root
RUN apt-get update && apt-get install -y gcc libpq-dev

USER airflow
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn==1.3.2 \
    joblib \
    pyyaml \
    dvc-s3

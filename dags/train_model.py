from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

def dvc_pull_data():
    os.system("dvc pull data/winequality-red.csv.dvc")

def train_and_save_model():
    with open("config/rf_config.yaml", "r") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/winequality-red.csv")

    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    with open("model/metrics.yaml", "w") as f:
        yaml.dump({"r2_score": float(r2), "rmse": float(rmse)}, f)

def dvc_add_model():
    os.system("dvc add model/model.pkl")
    os.system("dvc add model/metrics.yaml")

default_args = {
    "start_date": datetime(2024, 1, 1),
    "catchup": False
}

with DAG("train_model_daily",
         schedule_interval="@daily",
         default_args=default_args,
         tags=["mlops", "wine"],
         description="Train model daily and save to DVC") as dag:

    dvc_pull = PythonOperator(
        task_id="dvc_pull_data",
        python_callable=dvc_pull_data,
    )

    train_model = PythonOperator(
        task_id="train_and_save_model",
        python_callable=train_and_save_model,
    )

    dvc_track = PythonOperator(
        task_id="dvc_add_model",
        python_callable=dvc_add_model,
    )

    dvc_pull >> train_model >> dvc_track

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import subprocess
from pathlib import Path
import logging as lg

def dvc_pull_data():
    subprocess.run(
        ["dvc", "pull", "data/winequality-red.csv.dvc"],
        cwd="/app", 
        check=True
    )

def train_and_save_model():
    lg.info("📦 Загрузка параметров из config/rf_config.yaml...")
    base_path = Path("/app")
    config_path = base_path / "config" / "rf_config.yaml"
    data_path = base_path / "data" / "winequality-red.csv"
    model_dir = base_path / "model"
    model_file = model_dir / "model.pkl"
    metrics_file = model_dir / "metrics.yaml"

    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    lg.info("📥 Чтение данных из data/winequality-red.csv...")
    df = pd.read_csv(data_path)

    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lg.info("🧠 Обучение модели RandomForestRegressor...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=True)

    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        yaml.dump({"r2_score": float(r2), "rmse": float(rmse)}, f)

    lg.info(f"✅ Модель сохранена в: {model_file}")
    lg.info(f"📊 Метрики: R2 = {r2:.4f}, RMSE = {rmse:.4f}")


def dvc_add_model():
    base_path = Path("/app")
    model_file = base_path / "model" / "model.pkl"
    metrics_file = base_path / "model" / "metrics.yaml"

    lg.info("🗃️ Выполняется DVC add для артефактов модели...")

    try:
        subprocess.run(["dvc", "add", str(model_file)], cwd=base_path, check=True, capture_output=True, text=True)
        subprocess.run(["dvc", "add", str(metrics_file)], cwd=base_path, check=True, capture_output=True, text=True)
        lg.info("✅ DVC add выполнен успешно.")
    except subprocess.CalledProcessError as e:
        lg.error("❌ DVC add завершился с ошибкой:")
     

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

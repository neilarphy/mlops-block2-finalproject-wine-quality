import joblib
from pathlib import Path
import subprocess

MODEL_PATH = Path('model/model.pkl')


def pull_model():
    result = subprocess.run(
        ["dvc", "pull", str(MODEL_PATH) + ".dvc"], 
        check=True)
    return result.returncode == 0


def load_model():
    if not MODEL_PATH.exists():
        pull_model()
    return joblib.load(MODEL_PATH)

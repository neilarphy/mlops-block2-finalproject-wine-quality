import uvicorn
import logging
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
from sklearn import __version__ as sklearn_version

sys.path.append(str(Path(__file__).resolve().parent))
from model_utils import load_model
from predict import WineFeatures, make_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
lg = logging.getLogger(__name__)

model = None
MODEL_PATH = Path("model/model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    lg.info("Инициализация API и загрузка модели")
    try:
        if MODEL_PATH.exists():
            model = load_model()
            lg.info("Модель успешно загружена")
        else:
            lg.error("Модель не найдена")
            model = None
    except Exception as e:
        lg.error(f"Ошибка при загрузке модели: {e}")
        model = None
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(
    wine_features: WineFeatures
):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        result = make_prediction(model, wine_features)
        return {"predicted_quality": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthcheck")
async def healthcheck():
    if model is None:
        return {"status": "error", "reason": "Модель не загружена"}
    try:
        dummy = pd.DataFrame([{
            "fixed acidity": 7.4,
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        }])
        _ = model.predict(dummy)
        return {"status": "ok"}
    except Exception as e:
        return {
            "status": "error", 
            "reason": f"Предсказание не удалось с ошибкой {e}"}


@app.get("/model-info")
async def model_info():
    if model is None:
        return {"status": "error", "reason": "Модель не загружена"}
    return {
        "model_path": str(MODEL_PATH),
        "sklearn_version": sklearn_version,
        "features": [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
        ]
    }


if __name__ == "__main__":
    uvicorn.run('api:app', host="127.0.0.1", port=8000)

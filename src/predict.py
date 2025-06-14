from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity")
    volatile_acidity: float = Field(..., alias="volatile acidity")
    citric_acid: float = Field(..., alias="citric acid")
    residual_sugar: float = Field(..., alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide")
    density: float
    pH: float
    sulphates: float
    alcohol: float

    model_config = ConfigDict(populate_by_name=True)


def make_prediction(model, wine_features: WineFeatures) -> float:
    df = pd.DataFrame([wine_features.model_dump(by_alias=True)])
    prediction = model.predict(df)[0]
    return round(float(prediction), 2)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

app=FastAPI(title="House Price Intelligence API",version="1.0.0",
    description="Educational API using historical California census-area housing data, not current individual property listings.")
MODEL=Path(__file__).parent/"models"/"api_model.joblib"
FEATURES=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

def get_model():
    if MODEL.exists(): return joblib.load(MODEL)
    data_path=Path(__file__).parent/"data"/"california_housing.csv"
    if data_path.exists(): df=pd.read_csv(data_path)
    else:
        bunch=fetch_california_housing(as_frame=True); df=bunch.frame.copy()
    X=df[FEATURES]; y=df["MedHouseVal"]
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    model.fit(X,y); MODEL.parent.mkdir(exist_ok=True); joblib.dump(model,MODEL)
    return model

class PredictionInput(BaseModel):
    MedInc: float = Field(gt=0, le=50)
    HouseAge: float = Field(ge=1, le=100)
    AveRooms: float = Field(gt=0, le=100)
    AveBedrms: float = Field(gt=0, le=50)
    Population: float = Field(gt=0, le=1000000)
    AveOccup: float = Field(gt=0, le=100)
    Latitude: float = Field(ge=32, le=42.5)
    Longitude: float = Field(ge=-125, le=-114)

@app.get("/")
def root():
    return {"service":"House Price Intelligence API","docs":"/docs","note":"Historical California census-area median values, not current property valuations."}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload:PredictionInput):
    try:
        model=get_model()
        x=pd.DataFrame([payload.model_dump()])[FEATURES]
        value=float(model.predict(x)[0])*100000
        return {"predicted_historical_area_median_usd":round(value,2),"model":"RandomForestRegressor","warning":"Not a current individual property valuation."}
    except Exception as exc:
        raise HTTPException(status_code=500,detail=f"Prediction failed: {type(exc).__name__}")

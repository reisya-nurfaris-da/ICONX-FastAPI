from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

scaler = load_pickle("scaler.pkl")
model = load_pickle("model.pkl")

class PredictionRequest(BaseModel):
    Lat: float
    Long: float
    FAT_Port_to_Customers: float = Field(..., alias="FAT Port to Customers")
    Signal_OPM_ONT_dBm: float = Field(..., alias="Signal OPM ONT (dBm)")
    Mitra_AFB: float
    Mitra_IDM: float
    Mitra_IFT: float
    Mitra_INTENS: float
    Service_10: int
    Service_20: int
    Service_35: int
    Service_50: int
    dispo_dayofweek: int
    dispo_is_weekend: int

FEATURES = [
    "Lat", "Long", "FAT Port to Customers", "Signal OPM ONT (dBm)",
    "Mitra_AFB", "Mitra_IDM", "Mitra_IFT", "Mitra_INTENS", 
    "Service_10", "Service_20", "Service_35", "Service_50", 
    "dispo_dayofweek", "dispo_is_weekend"
]

app = FastAPI()

@app.post("/predict")
def predict(payload: PredictionRequest):
    input_dict = payload.dict(by_alias=True)
    try:
        data = [input_dict[feat] for feat in FEATURES]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in request: {e}")

    X = np.array(data).reshape(1, -1)

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaler transformation error: {e}")

    try:
        pred = model.predict(X_scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return {"prediction": float(pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from ml_pipeline import load_model, predict_comment, label_map

app = FastAPI(
    title="Toxic Comment Detection API",
    description="Multi-label toxicity detection for English + Russian comments.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str


class ToxicType(BaseModel):
    code: str
    name: str
    prob: float


class PredictResponse(BaseModel):
    text: str
    label: str
    overall_prob: float
    toxic_types: List[ToxicType]


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, overall_prob, toxic_types_raw = predict_comment(req.text, visualize=False)

    toxic_types = [
        ToxicType(code=code, name=label_map.get(code, code), prob=prob)
        for code, prob in toxic_types_raw
    ]

    return PredictResponse(
        text=req.text,
        label=label,
        overall_prob=overall_prob,
        toxic_types=toxic_types,
    )

# Put the code for your API here.
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# Load model and encoders
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }
    }


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: InputData):
    input_df = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }])

    from starter.ml.data import process_data
    from starter.ml.model import inference

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    result = lb.inverse_transform(preds)[0]

    return {"prediction": result}
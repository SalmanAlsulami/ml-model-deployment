# ML Model Deployment with FastAPI

This project builds a classification model using Census Bureau data and deploys it as a REST API using FastAPI on Render.

## Setup

Using pip and venv:
```bash
python3.13 -m venv .venv
.venv\Scripts\activate
pip install -r starter/requirements.txt
```

## Data

The dataset is `census.csv` located in `starter/data/`. The data contains some whitespace that gets cleaned automatically when running the training script.

## Model

The model is a Random Forest Classifier trained to predict whether income exceeds $50K/year. To train the model:
```bash
cd starter/starter
python train_model.py
```

This saves the model and encoders in `starter/model/` and outputs slice performance to `slice_output.txt`.

## API

Run the API locally:
```bash
cd starter
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` to test the endpoints.

## Tests
```bash
pytest starter/starter/test_model.py starter/starter/test_api.py -v
```

## Live API

The API is deployed at `https://ml-model-deployment-cq7q.onrender.com`

To test the live API:
```bash
cd starter
python live_post.py
```

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

DIR = os.path.dirname(os.path.abspath(__file__))

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


@pytest.fixture
def data():
    df = pd.read_csv(os.path.join(DIR, "..", "data", "census.csv"))
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df


@pytest.fixture
def trained_model(data):
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    return model, X_train, y_train, encoder, lb


def test_train_model_returns_random_forest(trained_model):
    """Test that train_model returns a RandomForestClassifier."""
    model, _, _, _, _ = trained_model
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_correct_shape(trained_model):
    """Test that inference returns predictions with correct shape."""
    model, X_train, _, _, _ = trained_model
    preds = inference(model, X_train)
    assert preds.shape[0] == X_train.shape[0]


def test_compute_model_metrics_returns_valid_values(trained_model):
    """Test that metrics are between 0 and 1."""
    model, X_train, y_train, _, _ = trained_model
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_process_data_returns_correct_types(data):
    """Test that process_data returns numpy arrays."""
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
# Script to train machine learning model.

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load data
data = pd.read_csv("../data/census.csv")

# Clean data - remove spaces from column names and values
data.columns = data.columns.str.strip()

data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# Split data
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Process training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {fbeta:.4f}")

# Save model and encoders
with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("../model/lb.pkl", "wb") as f:
    pickle.dump(lb, f)

print("Model saved successfully!")

# Slice performance
def compute_slice_metrics(df, feature, model, encoder, lb, cat_features):
    results = []
    for value in df[feature].unique():
        slice_df = df[df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        results.append(
            f"{feature}={value}: Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={fbeta:.4f}"
        )
    return results

# Output slice metrics to file
with open("slice_output.txt", "w") as f:
    for feature in cat_features:
        lines = compute_slice_metrics(
            test, feature, model, encoder, lb, cat_features
        )
        for line in lines:
            f.write(line + "\n")
        f.write("\n")

print("Slice metrics saved to slice_output.txt!")
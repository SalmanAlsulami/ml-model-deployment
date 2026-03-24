# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained to predict whether an individual's
income exceeds $50,000 per year based on census data. The model was developed using
scikit-learn's RandomForestClassifier with 100 estimators and a random state of 42.

## Intended Use
This model is intended for educational purposes to demonstrate machine learning
deployment using FastAPI and CI/CD pipelines. It can be used to predict income
categories based on demographic and employment features from census data.

## Training Data
The model was trained on the UCI Census Bureau dataset (census.csv) containing
32,561 records. The data was split 80/20 for training and testing. Categorical
features were encoded using OneHotEncoder and the label was binarized using
LabelBinarizer. All whitespace was stripped from the data before training.

## Evaluation Data
The evaluation data consists of 20% of the original census dataset, approximately
6,512 records, held out during the train-test split with random state 42.

## Metrics
The model was evaluated using Precision, Recall, and F1 Score:
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

## Ethical Considerations
The dataset contains sensitive demographic features such as race, sex, and
native country. These features may introduce bias into the model's predictions.
The model should not be used for making real decisions about individuals as it
may produce biased results against certain demographic groups.

## Caveats and Recommendations
- The model was trained on data from the 1990s and may not reflect current
  income distributions.
- Performance varies across demographic slices, as shown in slice_output.txt.
- It is recommended to retrain the model on more recent and representative data
  before any real-world application.
- Users should review the slice performance metrics to understand potential
  disparities across different demographic groups.
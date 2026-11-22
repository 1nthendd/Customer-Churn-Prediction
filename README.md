# Customer Churn Prediction

A machine learning project for predicting whether a telecom customer is likely to churn. The analysis uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and compares Logistic Regression, Random Forest, and CatBoost models.

The project is implemented as a Kaggle notebook. It covers data cleaning, exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, probability-threshold selection, and feature interpretation.

## Overview

Customer churn is a binary classification problem:

- **Yes** — the customer left the service
- **No** — the customer remained with the service

The goal is not only to obtain good overall accuracy, but also to identify customers at risk of leaving so that a retention team can take action.

## Notebook

[`customer-churn-prediction (5).ipynb`](customer-churn-prediction%20(5).ipynb) contains the complete analysis.

## Dataset

The project uses the **Telco Customer Churn** dataset:

- `customerID` — unique customer identifier
- `gender`, `Partner`, `Dependents` — customer information
- `tenure` — number of months with the company
- `PhoneService`, `MultipleLines`, `InternetService`, `Contract` — service attributes
- `MonthlyCharges`, `TotalCharges` — billing information
- `Churn` — target variable

## Project Workflow

### 1. Data quality checks

- Inspected the dataset structure and data types
- Checked for duplicate records and missing values
- Converted `TotalCharges` to numeric format
- Replaced missing `TotalCharges` values with `0`

### 2. Exploratory data analysis

The notebook explores:

- Contract type and churn
- Customer tenure
- Monthly and total charges
- Churn rates across categorical features
- Correlations between numerical features

### 3. Feature preparation

- Removed `customerID` and `Churn` from the feature matrix
- Used an 80/20 stratified train-test split
- Applied `StandardScaler` to numerical features
- Applied `OneHotEncoder` to categorical features
- Used a scikit-learn `ColumnTransformer` and `Pipeline` to keep preprocessing reproducible

### 4. Model training

The notebook compares:

1. **Logistic Regression** — interpretable baseline
2. **Random Forest** — nonlinear ensemble model with balanced class weights
3. **CatBoost** — gradient boosting model designed for categorical features

Model evaluation uses accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices.

### 5. Hyperparameter tuning

CatBoost is tuned with a small grid over:

- Tree depth: `4`, `6`, `8`
- Learning rate: `0.03`, `0.05`
- L2 regularization: `3`, `5`

Each configuration is evaluated by cross-validation using the same scoring metrics.

### 6. Classification threshold

Instead of relying only on the default probability threshold of `0.50`, the notebook calculates out-of-fold probabilities and evaluates thresholds from `0.10` to `0.70`.

The final CatBoost predictions use a threshold of **0.37**, with the goal of improving recall for the churn class while maintaining precision and F1 score.

## Baseline Results

The Logistic Regression baseline achieves:

| Metric | Score |
|---|---:|
| Accuracy | 0.806 |
| Precision (`Yes`) | 0.657 |
| Recall (`Yes`) | 0.559 |
| F1 score (`Yes`) | 0.604 |
| ROC-AUC | 0.842 |

The baseline confusion matrix contains:

- **926** true negatives
- **209** true positives
- **109** false positives
- **165** false negatives

For churn prediction, recall is especially important: false negatives represent customers who may leave without being identified.

## Feature Importance

The final CatBoost model exposes native feature importance scores. The notebook ranks and plots the top features, making the model's decisions easier to interpret from the business perspective.

## Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `catboost`
- `kagglehub`

## Running the Notebook

### Kaggle

1. Open the notebook in Kaggle.
2. Add the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) as an input dataset.
3. Run all cells.

The notebook is configured for the Kaggle input path:

```text
/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Local environment

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost kagglehub
jupyter notebook
```

Update the CSV path in the first code cell if running outside Kaggle.

## Possible Improvements

- Compare additional models such as XGBoost or LightGBM
- Perform broader hyperparameter optimization
- Test class-balancing techniques such as SMOTE or class-weight tuning
- Evaluate the final threshold with an independent validation process
- Use a retention-focused cost matrix for precision-recall trade-offs
- Add saved model and prediction artifacts
- Deploy the model as an API or interactive dashboard

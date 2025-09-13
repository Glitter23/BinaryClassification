## Overview

This repository contains a binary classification workflow for predicting customer churn using a Jupyter notebook (`BinaryPred.ipynb`). The workflow loads provided datasets, performs feature engineering on categorical variables, trains several scikit-learn models, evaluates them, and exports predictions.

### Repository contents
- `BinaryPred.ipynb`: End-to-end notebook for data preparation, modeling, evaluation, and export
- `train_PDjVQMB.csv`: Labeled training data
- `test_lTY72QC.csv`: Unlabeled test data for inference
- `solution.csv`: Sample predictions file produced by the notebook (KNN example)
- `solution_RANDOMFOR.csv`: Sample predictions file produced by the notebook (RandomForest example)

## Environment

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Packages used:
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install minimal dependencies:
```bash
pip install pandas numpy scikit-learn jupyter
```

## Data schema (expected columns)

The notebook expects at least the following columns to be present.
- `ID`: Unique identifier per row (used when exporting predictions)
- `Is_Churn`: Binary target label (available in `train_*.csv` only)
- `Age`: Numeric
- `Gender`: Categorical
- `Income`: Categorical
- `Balance`: Numeric
- `Vintage`: Numeric
- `Transaction_Status`: Numeric or categorical (treated as numeric in the notebook)
- `Product_Holdings`: Categorical
- `Credit_Card`: Numeric or boolean-like
- `Credit_Category`: Categorical

## Pipeline summary

1. Load `train_PDjVQMB.csv` and `test_lTY72QC.csv` using `pandas.read_csv`
2. Convert categorical columns to category dtype and create integer code columns using `Series.cat.codes`:
   - `Gender` → `gender_code`
   - `Income` → `Income_code`
   - `Product_Holdings` → `ProductHoldings_code`
   - `Credit_Category` → `CreCat_code`
3. Define the feature set used for modeling:
   - `features = ['Age','gender_code','Income_code','Balance','Vintage','Transaction_Status','ProductHoldings_code','Credit_Card','CreCat_code']`
4. Build `X_train`, `y_train` from `train` and `X_test` from `test`
5. Train one or more models (Logistic Regression, Random Forest, SVC, KNN)
6. Evaluate predictions against a pseudo `y_test` derived from the head of the training labels for reporting purposes
7. Export predictions with columns `[ID, Is_Churn]` to CSV

## Public usage (APIs/components in this repo)

There are no user-defined functions or classes. The public “API” of this repository is the notebook workflow itself and the generated outputs. Below are concrete usage examples that mirror the notebook steps for each model.

### Feature engineering example
```python
import pandas as pd
import numpy as np

train = pd.read_csv('train_PDjVQMB.csv')
test = pd.read_csv('test_lTY72QC.csv')

# Categorical encodings
train['gender_code'] = pd.Categorical(train['Gender']).codes
train['Income_code'] = pd.Categorical(train['Income']).codes
train['ProductHoldings_code'] = pd.Categorical(train['Product_Holdings']).codes
train['CreCat_code'] = pd.Categorical(train['Credit_Category']).codes

test['gender_code'] = pd.Categorical(test['Gender']).codes
test['Income_code'] = pd.Categorical(test['Income']).codes
test['ProductHoldings_code'] = pd.Categorical(test['Product_Holdings']).codes
test['CreCat_code'] = pd.Categorical(test['Credit_Category']).codes

features = ['Age','gender_code','Income_code','Balance','Vintage','Transaction_Status','ProductHoldings_code','Credit_Card','CreCat_code']

X_train = np.array(train[features])
y_train = np.array(train['Is_Churn'])
X_test = np.array(test[features])

# Pseudo y_test to enable metrics reporting (length-matched subset of y_train)
y_test = y_train[:len(X_test)]
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import f1_score

reg_log = LogisticRegression()
reg_log.fit(X_train, y_train)
y_pred = reg_log.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred, average='macro'))
```

### Random Forest (with export)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred, average='macro'))

# Export predictions
import pandas as pd
pred_df = pd.DataFrame({'ID': test['ID'], 'Is_Churn': y_pred})
pred_df.to_csv('solution_RANDOMFOR.csv', index=False)

# Feature importances
feature_df = pd.DataFrame({'Importance': reg_rf.feature_importances_, 'Features': features})
print(feature_df)
```

### Support Vector Classifier (SVC)
```python
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import f1_score

reg_svc = SVC()
reg_svc.fit(X_train, y_train)
y_pred = reg_svc.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred, average='macro'))
```

### K-Nearest Neighbors (with export)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import f1_score

reg_knn = KNeighborsClassifier()
reg_knn.fit(X_train, y_train)
y_pred = reg_knn.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred))

# Export predictions
import pandas as pd
pred_df = pd.DataFrame({'ID': test['ID'], 'Is_Churn': y_pred})
pred_df.to_csv('solution.csv', index=False)
```

## Running the notebook end-to-end

- Launch Jupyter and open `BinaryPred.ipynb`, then run all cells in order.
- Alternatively, execute the notebook non-interactively:
```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute BinaryPred.ipynb --output BinaryPred.out.ipynb
```

## Notes and caveats

- The evaluation shown in the notebook uses a pseudo `y_test` derived from the training labels solely to print metrics with the test predictions. This is not a proper validation strategy. Consider splitting the training set (e.g., `train_test_split`) or using cross-validation for meaningful evaluation.
- Ensure categorical encodings are applied consistently to both train and test before training.
- You can tune model hyperparameters (e.g., `RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)`) to improve performance.

## Outputs

- `solution_RANDOMFOR.csv`: Predictions from the Random Forest example with columns `[ID, Is_Churn]`
- `solution.csv`: Predictions from the KNN example with columns `[ID, Is_Churn]`
# AutoML Binary Classification

This AutoML system is created for work with tabular data (only with csv format).
You can make binary classification by using several algorithms as:
- Logistic Regression
- Naive Bayes
- SVM
- kNN
- Decision Tre
- Random Forest
- XGBoost

You can fit the model by all of this algorithms and system will choose the better one for yor metric.
There are 3 metrics:
- accuracy
- precision
- f1 score

By default system use all algorithms for fitting and accuracy metric

This system use sklearn library for all algorithms and metrics (except xgboost).

# Example

As an example -- heart.csv dataset in dataset directory.
Details about this dataset you can find [here](https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv).

System has interface with `fit` and `predict` methods.
## `fit`
```python
from binclass import BC


bc = BC()
bc.load_data_csv("dataset/heart.csv", "target")
bc.fit()
```
AutoML `fit` will print 
```py
Fitting by Logistic Regression
Fitting by Naive Bayes
Fitting by SVM
Fitting by kNN
Fitting by Decision Tree
Fitting by Random Forest
Fitting by XGBoost
Best algorithm is: Random Forest score is: 86.44
```
For this dataset optimal algorithm is Random Forest with the accuracy score 86.44.

## `predict`
```python
import pandas as pd
from binclass import BC

data = pd.read_csv("dataset/heart_extra.csv")
bc.predict(data)
```
AutoML `predict` will print 
```py
Prediction is: [0 0 0 0 1 0 0 0 0 0 1]
```
 
# Installation

Install from source code:
```
git clone https://github.com/RJ0x001/automl_bc.git
```
Create virtual environment:
```
python3 -m venv venv
venv\Scripts\activate
```
Install requirements
```
cd automl_bc
pip install -r requirements.txt
```

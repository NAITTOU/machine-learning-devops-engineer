# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project implements a machine learning pipeline to identify customers at high risk of churning. The solution includes:

- Exploratory Data Analysis (EDA) of customer demographics and service usage patterns

- Feature engineering to create meaningful predictors

- Model training with logistic regression and random forest classifiers

- Model evaluation using precision, recall, and ROC metrics

- MLOps best practices including modular code, logging, and model persistence

The implementation helps businesses proactively implement retention strategies by identifying at-risk customers through predictive analytics.

## Files and data description

```
.
├── Guide.ipynb
├── Guide.ipynb:Zone.Identifier
├── README.md
├── README.md:Zone.Identifier
├── __pycache__
│   ├── churn_library.cpython-36.pyc
│   └── churn_script_logging_and_tests.cpython-36-pytest-7.0.1.pyc
├── churn_library.py
├── churn_library.py:Zone.Identifier
├── churn_notebook.ipynb
├── churn_notebook.ipynb:Zone.Identifier
├── churn_script_logging_and_tests.py
├── churn_script_logging_and_tests.py:Zone.Identifier
├── data
│   ├── bank_data.csv
│   └── bank_data.csv:Zone.Identifier
├── images
│   ├── eda
│   │   ├── Churn_distribution.png
│   │   ├── Customer_Age_distribution.png
│   │   ├── Marital_Status_distribution.png
│   │   ├── Total_Trans_Ct_distribution.png
│   │   └── correlation_heatmap.png
│   └── results
│       ├── lr_report.png
│       ├── rf_feature_importance.png
│       ├── rf_report.png
│       └── roc_curves.png
├── logs
│   └── results.log
├── models
│   ├── logistic_model.pkl
│   ├── logistic_model.pkl:Zone.Identifier
│   ├── rfc_model.pkl
│   └── rfc_model.pkl:Zone.Identifier
├── requirements_py3.10.txt
├── requirements_py3.10.txt:Zone.Identifier
├── requirements_py3.6.txt
├── requirements_py3.6.txt:Zone.Identifier
├── requirements_py3.8.txt
└── requirements_py3.8.txt:Zone.Identifier
```

### Dataset Features:

- CLIENTNUM: Unique client identifier

- Attrition_Flag: variable indicating customer status

- Customer_Age: Numeric value representing customer's age

- Gender: Biological sex (M/F)

- Dependent_count: Number of dependents/children associated

- Education_Level: Categorical education status (High School, Graduate, Uneducated, Unknown)

- Marital_Status: Relationship status category (Married, Single, Unknown)

- Income_Category: Annual income bracket
(Less than 40K,40K,60K – 80K,80K,80K – $120K)

## Running Files

1. Install Dependencies:
```
pip install -r requirements_py3.6.txt
```

2. Execute Full Pipeline:
```
python churn_library.py
```

This will:

- Perform data ingestion and preprocessing

- Generate EDA visualizations in images/eda/

- Train and evaluate models (outputs in images/results/)

- Save trained models to models/

- Log execution details to logs/result.log

3. Run Unit Tests:

```
python churn_script_logging_and_tests.py
```

You can run the unit test with pytest as well : 

Install pytest :

```
pip install pytest
```

Then run

```
pytest  churn_script_logging_and_tests.py
```



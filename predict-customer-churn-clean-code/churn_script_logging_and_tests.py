# unit test doc string
"""
This module contains necessary function to perfoem
unit testing for customer Chrun library
"""

import os
import logging
import pytest
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

data_frame = cl.import_data("./data/bank_data.csv")


@pytest.fixture
def import_data():
    '''
    Fixture to provide the import_data function from churn_library.
    '''
    yield cl.import_data


@pytest.fixture
def perform_eda():
    '''
    Fixture to provide the perform_eda function from churn_library.
    '''
    yield cl.perform_eda


@pytest.fixture
def encoder_helper():
    '''
    Fixture to provide the encoder_helper function from churn_library.
    '''
    yield cl.encoder_helper


@pytest.fixture
def perform_feature_engineering():
    '''
    Fixture to provide the perform_feature_engineering function from churn_library.
    '''
    yield cl.perform_feature_engineering


@pytest.fixture
def train_models():
    '''
    Fixture to provide the train_models function from churn_library.
    '''
    yield cl.train_models


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(data_frame)
        plots_list = [
            "Churn_distribution.png",
            "Customer_Age_distribution.png",
            "Marital_Status_distribution.png",
            "Total_Trans_Ct_distribution.png",
            "correlation_heatmap.png"
        ]
        for plot in plots_list:
            if not os.path.exists(f"./images/eda/{plot}"):
                raise FileNotFoundError(f"Plot {plot} not found")
            logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: ERROR - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        category_columns = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"
        ]
        encoded_df = encoder_helper(data_frame, category_columns, "Churn")
        encoded_cols_list = [f"{col}_Churn" for col in category_columns]
        columns_set = set(encoded_df.columns)
        for col in encoded_cols_list:
            assert col in columns_set
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: ERROR - Column %s not found", col)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_frame, "Churn")
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: "
            "ERROR - Sample size for train or test doesn't match target")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        # Perform feature engineering to get train/test data
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            data_frame, "Churn")

    # Train models
        train_models(x_train, x_test, y_train, y_test)

    # Check if models are saved
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")

    # Check if ROC curve and feature importance plots are saved
        assert os.path.exists("./images/results/roc_curves.png")
        assert os.path.exists("./images/results/rf_feature_importance.png")

    # Check if classification reports are saved
        assert os.path.exists("./images/results/rf_report.png")
        assert os.path.exists("./images/results/lr_report.png")
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: ERROR - %s", err)
        raise err


if __name__ == "__main__":
    # Run all tests
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)

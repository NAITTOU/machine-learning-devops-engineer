# library doc string
"""
This module contains necessary function to perfoem
model training that predict customer Chrun
"""

# import libraries
import os
import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        logging.info("SUCCESS: Data imported!")
        return data_frame
    except FileNotFoundError as err:
        logging.error("ERROR: csv file path is inccorect %s", err)
        return None


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # Create a new column 'Churn' based on 'Attrition_Flag'
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # List of columns to plot
    columns = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']

    # Generate plots for each column
    for column in columns:
        plt.figure(figsize=(20, 10))
        if column == 'Marital_Status':
            data_frame[column].value_counts(normalize=True).plot(kind='bar')
        elif column == 'Total_Trans_Ct':
            sns.histplot(data_frame[column], stat='density', kde=True)
        else:
            data_frame[column].hist()
        plt.title(f'{column} Distribution')
        plt.savefig(f'images/eda/{column}_distribution.png')
        plt.close()

    # Generate correlation heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('images/eda/correlation_heatmap.png')
    plt.close()
    logging.info(
        "SUCCESS: EDA reports saved to the 'images/eda' folder.")


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        # Calculate average churn rate for each category
        avg_churn = data_frame.groupby(category).mean()[response]

        # Create a list to store the churn rates for each row
        churn_list = []

        # Loop through the column and append the corresponding churn rate
        for val in data_frame[category]:
            churn_list.append(avg_churn.loc[val])

        # Add the new column to the dataframe
        data_frame[f'{category}_Churn'] = churn_list

    logging.info(
        "SUCCESS: categorical columns encoding completed!")

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_data = data_frame[response]
    x_data = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        f'Gender_{response}',
        f'Education_Level_{response}',
        f'Marital_Status_{response}',
        f'Income_Category_{response}',
        f'Card_Category_{response}']

    x_data[keep_cols] = data_frame[keep_cols]

    # This cell may take up to 15-20 minutes to run
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    logging.info(
        "SUCCESS: Data split to train and test data is completed!")

    return x_train, x_test, y_train, y_test


def classification_report_image(y_data):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_data:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Define the reports to generate as a dictionary of dictionaries

    y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = y_data

    reports = {
        'Random_Forest': {
            'title_train': 'Random Forest Train',
            'y_train': y_train,
            'y_train_preds': y_train_preds_rf,
            'title_test': 'Random Forest Test',
            'y_test': y_test,
            'y_test_preds': y_test_preds_rf,
            'filename': 'images/results/rf_report.png'
        },
        'Logistic_Regression': {
            'title_train': 'Logistic Regression Train',
            'y_train': y_train,
            'y_train_preds': y_train_preds_lr,
            'title_test': 'Logistic Regression Test',
            'y_test': y_test,
            'y_test_preds': y_test_preds_lr,
            'filename': 'images/results/lr_report.png'
        }
    }

    # Generate and save classification reports using a for loop
    for _, report in reports.items():
        plt.rc('figure', figsize=(5, 5))

        # Add the training report
        plt.text(0.01, 1.25, str(report['title_train']), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    report['y_train'], report['y_train_preds'])), {
                'fontsize': 10}, fontproperties='monospace')

        # Add the testing report
        plt.text(0.01, 0.6, str(report['title_test']), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    report['y_test'], report['y_test_preds'])), {
                'fontsize': 10}, fontproperties='monospace')

        plt.axis('off')
        plt.savefig(report['filename'])
        plt.close()

    logging.info(
        "SUCCESS: Classification reports saved to the 'images/results' folder.")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)
    plt.close()

    logging.info("SUCCESS: Feature importance plot saved to %s", output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Generate and save classification reports
    y_data = (y_train,
              y_test,
              y_train_preds_lr,
              y_train_preds_rf,
              y_test_preds_lr,
              y_test_preds_rf)
    classification_report_image(y_data)

    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # plots
    plt.figure(figsize=(15, 8))
    a_x = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=a_x,
        alpha=0.8)
    lrc_plot.plot(ax=a_x, alpha=0.8)
    plt.savefig('./images/results/roc_curves.png')
    plt.close()

    feature_importance_plot(
        cv_rfc,
        x_train,
        './images/results/rf_feature_importance.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    logging.info(
        "SUCCESS: Model training and evaluation completed. "
        "Results saved to './images/results' and models saved to './models'."
    )


if __name__ == "__main__":
    # Load data
    customer_data_frame = import_data('data/bank_data.csv')

    # Perform EDA
    perform_eda(customer_data_frame)

    # Encode categorical features
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    customer_data_frame = encoder_helper(
        customer_data_frame, category_list, 'Churn')

    # Perform feature engineering
    X_data_train, X_data_test, y_data_train, y_data_test = perform_feature_engineering(
        customer_data_frame, 'Churn')

    # Train models and save results
    train_models(X_data_train, X_data_test, y_data_train, y_data_test)

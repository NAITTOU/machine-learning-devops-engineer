import os
import logging
import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	plots_list= ["churn",
                 "Customer_Age",
                 "Marital_Status",
                 "Total_Trans_Ct",
                 "Heatmap"
                ]
	try:
		for plot in plots_list:
			with open("./images/{}".format(plot), "r"):
				logging.info("SUCCESS: imges opened successfuly")
	except FileNotFoundError as err:
		logging.error("error: images not found")
		raise err

	
        
        
            


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
    
	encoder_helper(df)
	try:
		encoded_cols_list = [
			"Gender_Churn",
			"Education_Level_Churn",
			"Marital_Status_Churn",
			"Income_Category_Churn",
			"Card_Category_Churn"
			
		]
        
		columns_set = set(encoded_df.columns)
		for col in encoded_cols_list:
			assert col in columns_set
	except AssertionError as err:
		logging.error(f"ERROR: colums {col} not found")
	
	

def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass









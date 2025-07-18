import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
from xgboost import XGBRegressor
import pyodbc

# old data - for comparison
data = pd.read_csv(r'S:\Finance & Performance\IM&T\BIReporting\Data science projects\CaseDuration - Theatre case duration prediction\Dataset.csv', encoding='ISO-8859-1')

# import new data
dsn = "coch_p2" 

read_connection = pyodbc.connect(f'DSN={dsn}', autocommit=True)

sql_query = "InformationSandpitDB.datascience.CaseDuration_predict"
df = pd.read_sql_query(sql_query, read_connection)

read_connection.close() # best practice to close the connection


# load trained model and pipeline
loaded_pipeline = joblib.load('xgboost_model_pipeline.joblib')
prediction = loaded_pipeline.predict(new_data)
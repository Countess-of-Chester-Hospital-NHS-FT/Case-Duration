import pandas as pd
import joblib
import pyodbc
from sqlalchemy import create_engine, text

# import data needing predictions (unbooked waitlist patients)
dsn = "coch_p2" 
read_connection = pyodbc.connect(f'DSN={dsn}', autocommit=True)
sql_query = "select * from InformationSandpitDB.datascience.CaseDuration_predict"
new_data = pd.read_sql_query(sql_query, read_connection)
read_connection.close()

# load trained model and pipeline
loaded_pipeline = joblib.load('final_best_model.joblib')
prediction = loaded_pipeline.predict(new_data)

# join results back to ids
results = pd.DataFrame({
    "id": new_data["ip_encntr_id"].values,
    "prediction": prediction,
    "prediction_date": pd.Timestamp.now()
})

# write predictions back to the warehouse, so that existing ids are overwritten but new ids are appended (upsert)
connection_params = f"DSN={dsn};DATABASE=InformationSandpitDB"
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={connection_params}')

# 1. Write predictions to a staging table
results.to_sql(
    name='CaseDuration_predictions_stage',
    con=engine,
    schema='datascience',
    if_exists='replace',  # drop/create staging table each run
    index=False
)

# 2. Run a merge SQL to upsert into the main table
merge_sql = """
MERGE datascience.CaseDuration_predictions AS target
USING datascience.CaseDuration_predictions_stage AS source
ON target.id = source.id
WHEN MATCHED THEN
    UPDATE SET prediction = source.prediction,
               prediction_date = source.prediction_date
WHEN NOT MATCHED THEN
    INSERT (id, prediction, prediction_date)
    VALUES (source.id, source.prediction, source.prediction_date);
"""

with engine.begin() as conn:
    conn.execute(text(merge_sql))

engine.dispose()
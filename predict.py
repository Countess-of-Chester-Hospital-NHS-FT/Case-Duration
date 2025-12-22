import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# change this manually when you want to update the model
model_version = "2025-10-17_kinkajou.joblib"

folder_path = os.environ.get('DS_FOLDER_PATH')
if not folder_path:
    raise ValueError("DS_FOLDER_PATH environment variable not set!")

model_path = os.path.join(folder_path, "Case Duration - Theatre case duration prediction","models", model_version)

# import data needing predictions (unbooked waitlist patients) and create a backup of the predictions table
dsn = "coch_p2"
connection_url_cerner = URL.create(
    "mssql+pyodbc",
    query={"odbc_connect": f"DSN={dsn};Database=CernerStaging"}
)
engine_cerner = create_engine(connection_url_cerner)

sql_query = "select * from InformationSandpitDB.datascience.CaseDuration_predict"
new_data = pd.read_sql_query(sql_query, engine_cerner)

backup_query = "select * from InformationSandpitDB.datascience.CaseDuration_predictions"
backup_data = pd.read_sql_query(backup_query, engine_cerner)
backup_path = os.path.join(folder_path, "Case Duration - Theatre case duration prediction","predictions_backup", "CaseDuration_predictions.pkl")
backup_data.to_pickle(backup_path)

engine_cerner.dispose()

# load trained model and pipeline
loaded_pipeline = joblib.load(model_path)
prediction = loaded_pipeline.predict(new_data)

# join results back to ids
results = pd.DataFrame({
    "id": new_data["ip_encntr_id"].values,
    "prediction": prediction,
    "prediction_date": pd.Timestamp.now(),
    "model_version": model_version
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
               prediction_date = source.prediction_date,
               model_version = source.model_version
WHEN NOT MATCHED THEN
    INSERT (id, prediction, prediction_date, model_version)
    VALUES (source.id, source.prediction, source.prediction_date, source.model_version);
"""

with engine.begin() as conn:
    conn.execute(text(merge_sql))

engine.dispose()

print("finished writing results")

with open("script_log.txt", "a") as f:
    f.write(f"Script ran at: {pd.Timestamp.now()}\n")
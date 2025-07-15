# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import Bunch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# import data
data = pd.read_csv(r'S:\Finance & Performance\IM&T\BIReporting\Data science projects\CaseDuration - Theatre case duration prediction\Dataset.csv', encoding='ISO-8859-1')

# categorise columns for pre-processing
categorical_low_cardinality = ['SessionType', 'WoundClass', 'PatientClassification', 'ASAScore', 'PatientSex']
categorical_high_cardinality = ['Specialty', 'ProcedureCategory', 'PrimaryProcedure', 'ConsultantSurgeon', 'OperatingSurgeon', 'FirstAssistant', 'Anaesthetist']
numerical_features = ['SecondaryProcedures', 'AgeAtCase', 'PatientBMI', 'SystolicBloodPressure', 'DiastolicBloodPressure', 'HeartRateMonitored', 'AvgDurationProc', 'AvgDurationProcAndSurgeon']
binary_features = ['LaproscopicFlag', 'COMORB_ThyroidProblems2', 'COMORB_Endocarditis', 'COMORB_HeartMurmur', 'COMORB_RheumaticFever', 'COMORB_KidneyDisease', 'COMORB_LiverDisease', 'COMORB_Diabetes', 'COMORB_Pacemaker', 'COMORB_Stroke', 'COMORB_AnaestheticProblems']

# define target and features
y = data['CaseDuration']
X = data[categorical_low_cardinality + categorical_high_cardinality + numerical_features + binary_features]

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create pipelines for pre-processing steps
def cap_at_95th_percentile(X):
    return np.apply_along_axis(lambda x: np.clip(x, None, np.percentile(x, 95)), axis=0, arr=X)

numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('cap', FunctionTransformer(cap_at_95th_percentile, validate=False, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())
    ]
)

categorical_low_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Not known')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

categorical_high_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Not known')),
        ('target_enc', TargetEncoder()),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat_low', categorical_low_transformer, categorical_low_cardinality),
        ('cat_high', categorical_high_transformer, categorical_high_cardinality),
        ('binary', 'passthrough', binary_features)
    ]
)

###### Test the pre processing works ####################
# Fit and pre-process the training data
X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)

# pre process the test data
X_test_preprocessed = preprocessor.transform(X_test)

# convert them back to dataframes with meaningful column names so I can inspect
feature_names = preprocessor.get_feature_names_out()
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)

rename_mapping = {
    f'cat_high__{i}': col_name
    for i, col_name in enumerate(categorical_high_cardinality)
}

X_train_preprocessed_df = X_train_preprocessed_df.rename(columns=rename_mapping)
####################################################################################

# Train the model
best_hyperparameters = {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.05, 'colsample_bytree': 1.0}
xgb_model = XGBRegressor(**best_hyperparameters)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ]
)

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Scoring metrics
def percentage_within_10(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)) <= 0.1)

def percent_overage(y_true, y_pred):
    epsilon = 1e-10
    return np.mean((y_pred - y_true) / (y_true + epsilon) > 0.1)

def percent_underage(y_true, y_pred):
    epsilon = 1e-10
    return np.mean((y_pred - y_true) / (y_true + epsilon) < -0.1)

# Compute evaluation metrics
mape = mean_absolute_percentage_error(y_test, y_pred)
pct_within_10 = percentage_within_10(y_test, y_pred)
pct_overage = percent_overage(y_test, y_pred)
pct_underage = percent_underage(y_test, y_pred)

print(f"""mape:{round(mape, 2)}
pct_within_10:{round(pct_within_10, 2)}
pct_overage:{round(pct_overage, 2)}
pct_underage:{round(pct_underage, 2)}""")

# Check for overfitting
# Plot errors
epsilon = 1e-10
percentage_error = (y_pred - y_test.values) / (y_test.values + epsilon)

percentage_error_percent = percentage_error * 100
    # Clip percentage errors to range -100% to +100%
percentage_error_percent = np.clip(percentage_error_percent, -100, 100)
    # Plot histogram
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.hist(
    percentage_error_percent,
    bins=50,
    range=(-100, 100),
    weights=np.ones_like(percentage_error_percent) / len(percentage_error_percent),
    edgecolor='black'
    )
ax.set_xlim(-100, 100)
ax.set_xlabel('Percentage Error (%)')
ax.set_ylabel('Proportion of Predictions')
ax.set_title(f'Distribution of Errors')
plt.show()


# Save and export the fitted model
model_filename = 'xgboost_model_pipeline.joblib'
joblib.dump(pipeline, model_filename)

print(f"Model pipeline saved to {model_filename}")
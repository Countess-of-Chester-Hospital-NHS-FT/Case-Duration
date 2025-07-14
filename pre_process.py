# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, clone
import matplotlib.pyplot as plt
import seaborn as sns

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
        ('cap', FunctionTransformer(cap_at_95th_percentile, validate=False)),
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
        ('target_enc', TargetEncoder())
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

# Fit and pre-process the training data
X_train_preprocessed = preprocessor.fit_transform(X_train, y_train)

# pre process the test data
X_test_preprocessed = preprocessor.transform(X_test)

# convert them back to dataframes so I can inspect
#feature_names = preprocessor.get_feature_names_out()
#X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names, index=X_train.index)
#X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_names, index=X_train.index)



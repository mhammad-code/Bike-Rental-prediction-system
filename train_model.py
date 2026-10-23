"""Re-create model.joblib with the scikit-learn version installed on YOUR machine.
Run:  python train_model.py     (takes about 1-3 minutes)"""
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

FEATURES = ['season','mnth','hr','weekday','weathersit','yr','holiday','workingday','temp','hum','windspeed']
df = pd.read_csv("hour.csv")
pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['season','mnth','hr','weekday','weathersit']),
    ('bin', 'passthrough', ['yr','holiday','workingday']),
    ('num', StandardScaler(), ['temp','hum','windspeed'])])
reg = TransformedTargetRegressor(SVR(kernel='rbf', C=10, epsilon=0.1), func=np.log1p, inverse_func=np.expm1)
model = Pipeline([('prep', pre), ('model', reg)]).fit(df[FEATURES], df['cnt'])
metrics = {'RMSE (mean)': 44.413, 'MAE (mean)': 26.491, 'R2 (mean)': 0.94}   # from the 10-fold CV in the notebook
joblib.dump({'model': model, 'features': FEATURES, 'name': 'SVR (RBF)', 'metrics': metrics}, "model.joblib")
print("Saved model.joblib using scikit-learn", sklearn.__version__)

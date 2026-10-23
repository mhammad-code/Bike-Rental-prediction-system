import os, datetime
import numpy as np, pandas as pd, joblib, streamlit as st

st.set_page_config(page_title="Bike Sharing Demand Predictor", page_icon="🚲", layout="centered")

FEATURES = ['season','mnth','hr','weekday','weathersit','yr','holiday','workingday','temp','hum','windspeed']

def build_and_train():
    """Train the SVR (RBF) pipeline on hour.csv with the scikit-learn version installed here."""
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    df = pd.read_csv("hour.csv")
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['season','mnth','hr','weekday','weathersit']),
        ('bin', 'passthrough', ['yr','holiday','workingday']),
        ('num', StandardScaler(), ['temp','hum','windspeed'])])
    reg = TransformedTargetRegressor(SVR(kernel='rbf', C=10, epsilon=0.1), func=np.log1p, inverse_func=np.expm1)
    model = Pipeline([('prep', pre), ('model', reg)]).fit(df[FEATURES], df['cnt'])
    return model

@st.cache_resource(show_spinner="Preparing the model (first run only, may take 1-3 minutes)...")
def load_model():
    """Load model.joblib and test it. If it fails to load OR to predict (e.g. it was saved with a
    different scikit-learn version), retrain the same SVR model locally and cache it."""
    probe = pd.DataFrame([{'season':3,'mnth':7,'hr':17,'weekday':3,'weathersit':1,'yr':1,
                           'holiday':0,'workingday':1,'temp':0.7,'hum':0.5,'windspeed':0.2}])[FEATURES]
    try:
        bundle = joblib.load("model.joblib")
        bundle["model"].predict(probe)          # smoke test
        return bundle["model"], bundle["name"], bundle.get("metrics", {})
    except Exception:
        model = build_and_train()
        try:
            joblib.dump({'model': model, 'features': FEATURES, 'name': 'SVR (RBF)', 'metrics': {}}, "model.joblib")
        except Exception:
            pass
        return model, "SVR (RBF, retrained on this machine)", {}

model, model_name, metrics = load_model()

st.title("🚲 Bike Sharing Demand Predictor")
st.caption("Predict hourly bike rentals for Capital Bikeshare (Washington D.C.) from date, time and weather.")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date", datetime.date(2012, 7, 10), min_value=datetime.date(2011, 1, 1), max_value=datetime.date(2030, 12, 31))
    hour = st.slider("Hour of day", 0, 23, 17)
    holiday = st.checkbox("Public holiday")
    weather = st.selectbox("Weather", ["Clear / partly cloudy", "Mist / cloudy", "Light rain / light snow", "Heavy rain / snow / fog"])
with col2:
    temp_c = st.slider("Temperature (°C)", -8.0, 41.0, 25.0, 0.5)
    hum = st.slider("Humidity (%)", 0, 100, 55)
    wind = st.slider("Wind speed (km/h)", 0, 67, 12)

month = date.month
season = 1 if month in (3,4,5) else 2 if month in (6,7,8) else 3 if month in (9,10,11) else 4
weekday = (date.weekday() + 1) % 7            # dataset: 0 = Sunday
workingday = int(weekday not in (0, 6) and not holiday)
yr = 1 if date.year >= 2012 else 0

row = pd.DataFrame([{
    'season': season, 'mnth': month, 'hr': hour, 'weekday': weekday,
    'weathersit': ["Clear / partly cloudy", "Mist / cloudy", "Light rain / light snow", "Heavy rain / snow / fog"].index(weather) + 1,
    'yr': yr, 'holiday': int(holiday), 'workingday': workingday,
    'temp': temp_c / 41, 'hum': hum / 100, 'windspeed': wind / 67}])[FEATURES]

if st.button("Predict rentals", type="primary", use_container_width=True):
    pred = max(0, int(round(float(model.predict(row)[0]))))
    st.metric("Predicted bike rentals this hour", f"{pred:,}")
    # whole-day profile for the chosen conditions
    day = pd.concat([row.assign(hr=h) for h in range(24)], ignore_index=True)
    profile = pd.Series(np.clip(model.predict(day), 0, None).round(), index=range(24), name="Predicted rentals")
    st.subheader("Predicted demand across the day")
    st.bar_chart(profile)

with st.expander("About the model"):
    st.write(f"**Algorithm:** {model_name}")
    if metrics:
        st.write(f"10-fold CV on the full dataset: RMSE ≈ {metrics.get('RMSE (mean)', float('nan')):.1f}, "
                 f"MAE ≈ {metrics.get('MAE (mean)', float('nan')):.1f}, R² ≈ {metrics.get('R2 (mean)', float('nan')):.3f}")
    st.write("Dataset: UCI Bike Sharing (2011–2012, 17,379 hourly records). Temperature, humidity and wind are normalised internally.")

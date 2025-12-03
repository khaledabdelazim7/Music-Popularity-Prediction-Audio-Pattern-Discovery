import streamlit as st
import pickle
import pandas as pd

# ============================================
# LOAD MODELS
# ============================================
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


feature_cols = [
    'artists', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'time_signature', 'track_genre', 'energy_ratio_danceability',
    'duration_ms_log', 'popularity_density', 'popularity_weight'
]

numeric_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'energy_ratio_danceability', 'duration_ms_log',
    'popularity_density', 'popularity_weight'
]

categorical_features = ['artists', 'explicit', 'track_genre']

# ============================================
# STREAMLIT UI
# ============================================
st.title("🎵 XGBoost Popularity Prediction")

st.write("Enter features to get a popularity prediction.")

user_vals = {}

st.subheader("Categorical (Label Encoded)")
for col in categorical_features:
    user_vals[col] = st.number_input(col, step=1, value=0)

st.subheader("Numeric Features")
for col in numeric_features:
    user_vals[col] = st.number_input(col, value=0.0)

# Convert to DataFrame
df_input = pd.DataFrame([user_vals])

# ============================================
# FIX COLUMN ORDER BEFORE PREDICTION
# ============================================
df_input = df_input[feature_cols]   # <--- THE MOST IMPORTANT FIX

# ============================================
# PREDICT
# ============================================
if st.button("Predict Popularity"):
    try:
        # Scale numeric columns
        df_input[numeric_features] = scaler.transform(df_input[numeric_features])

        # Predict
        pred = xgb_model.predict(df_input)[0]

        st.success(f"🎯 Predicted Popularity: {pred:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import firebase_admin
from firebase_admin import credentials, firestore

# ----------------- FIREBASE SETUP -----------------
firebase_config = st.secrets["firebase"]  # from secrets.toml
cred = credentials.Certificate(dict(firebase_config))

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ----------------- LOAD / TRAIN MODEL -----------------
MODEL_FILE = "model.pkl"
ENC_FILE = "encoders.pkl"
FEATURE_FILE = "feature_columns.pkl"

if not (os.path.exists(MODEL_FILE) and os.path.exists(ENC_FILE) and os.path.exists(FEATURE_FILE)):
    st.info("🔄 Training model (first run)... Please wait.")

    # Load dataset
    df = pd.read_csv("adult.csv")

    target = "income"
    categorical_cols = [
        "workclass", "education", "marital.status", "occupation",
        "relationship", "race", "sex", "native.country"
    ]

    # Encode categoricals
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])

    # Train/test split
    X = df.drop(columns=[target, "fnlwgt"])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model, encoders, and feature list
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENC_FILE, "wb") as f:
        pickle.dump({"encoders": encoders, "target_encoder": target_encoder}, f)
    with open(FEATURE_FILE, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    st.success("✅ Model trained and saved!")

# Load saved model + encoders
model = pickle.load(open(MODEL_FILE, "rb"))
enc_data = pickle.load(open(ENC_FILE, "rb"))
feature_columns = pickle.load(open(FEATURE_FILE, "rb"))

encoders = enc_data["encoders"]
target_encoder = enc_data["target_encoder"]

# ----------------- STREAMLIT UI -----------------
st.markdown(
    "<h1 style='text-align: center;'>Census Income Prediction App</h1>",
    unsafe_allow_html=True,
)
st.write("### Enter Census Details:")

with st.form("prediction_form", clear_on_submit=True):
    age = st.number_input("Age", 18, 100, 30, key="age")
    workclass = st.selectbox("Workclass", encoders["workclass"].classes_, key="workclass")
    education = st.selectbox("Education", encoders["education"].classes_, key="education")
    marital_status = st.selectbox("Marital Status", encoders["marital.status"].classes_, key="marital")
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_, key="occupation")
    relationship = st.selectbox("Relationship", encoders["relationship"].classes_, key="relationship")
    race = st.selectbox("Race", encoders["race"].classes_, key="race")
    sex = st.selectbox("Sex", encoders["sex"].classes_, key="sex")
    hours_per_week = st.number_input("Hours per week", 1, 100, 40, key="hours")
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0, key="gain")
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0, key="loss")
    native_country = st.selectbox("Native Country", encoders["native.country"].classes_, key="country")

    submitted = st.form_submit_button("Predict Income")

if submitted:
    # Encode inputs
    data = {
        "age": age,
        "workclass": encoders["workclass"].transform([workclass])[0],
        "education": encoders["education"].transform([education])[0],
        "marital.status": encoders["marital.status"].transform([marital_status])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "relationship": encoders["relationship"].transform([relationship])[0],
        "race": encoders["race"].transform([race])[0],
        "sex": encoders["sex"].transform([sex])[0],
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": encoders["native.country"].transform([native_country])[0],
    }

    # Reindex input to match training features
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(input_df)[0]
    income = target_encoder.inverse_transform([pred])[0]

    st.markdown(
        f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px; text-align: center;'>"
        f"<b>Predicted Income: {income}</b></div>",
        unsafe_allow_html=True,
    )

    # Save to Firestore
    db.collection("predictions").add({
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "native_country": native_country,
        "predicted_income": income
    })
    st.info("✅ Prediction saved to Firestore")

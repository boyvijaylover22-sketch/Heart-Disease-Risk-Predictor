import math
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_model.pkl"


def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error("Model file not found. The app cannot run without heart_disease_model.pkl.")
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load the model: {exc}")
        st.stop()


bundle = load_model_bundle()
model_type = bundle.get("model_type", "sklearn") if isinstance(bundle, dict) else "sklearn"
feature_names = bundle.get("features") if isinstance(bundle, dict) else None
model = bundle.get("model") if isinstance(bundle, dict) else None

if feature_names is None:
    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]


def score_with_rule_based_model(values):
    weights = bundle.get("weights", {}) if isinstance(bundle, dict) else {}
    intercept = float(bundle.get("intercept", -5.0)) if isinstance(bundle, dict) else -5.0
    score = intercept
    for feature_name in feature_names:
        if feature_name in weights:
            score += float(weights[feature_name]) * float(values.get(feature_name, 0))
    probability = 1 / (1 + math.exp(-score))
    return probability


st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter the clinical values below to estimate the likelihood of heart disease.")

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 55)
        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda value: "Female" if value == 0 else "Male",
        )
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            [1, 2, 3, 4],
            format_func=lambda value: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic",
            }[value],
        )
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
        )

    with col2:
        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda value: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy",
            }[value],
        )
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda value: "No" if value == 0 else "Yes",
        )
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [1, 2, 3],
            format_func=lambda value: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping",
            }[value],
        )
        ca = st.number_input("Number of Major Vessels", 0, 4, 0)
        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
            format_func=lambda value: {
                3: "Normal",
                6: "Fixed defect",
                7: "Reversible defect",
            }[value],
        )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    if model_type == "rule_based":
        probability = score_with_rule_based_model(values)
        risk_label = "High risk" if probability >= 0.5 else "Lower risk"
    else:
        row = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        probability = model.predict_proba([row])[0][1]
        risk_label = "High risk" if probability >= 0.5 else "Lower risk"

    st.subheader("Prediction")
    st.metric("Risk Probability", f"{probability * 100:.1f}%")

    if risk_label == "High risk":
        st.error("⚠️ The model predicts a high likelihood of heart disease.")
    else:
        st.success("✅ The model predicts a lower likelihood of heart disease.")

    st.caption("This estimate is based on a trained machine-learning model and should be used as a guide, not a diagnosis.")
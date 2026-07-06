from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

# --------------------------------------------------
# FORM FIELD DEFINITIONS
# --------------------------------------------------
FIELD_DEFINITIONS: List[Dict[str, Any]] = [
    {"name": "age", "label": "Age", "input_type": "number", "min": 0, "max": 120, "step": 1, "default": 55},
    {"name": "sex", "label": "Sex", "input_type": "select", "options": [(0, "Female"), (1, "Male")], "default": 1},
    {"name": "cp", "label": "Chest Pain Type (cp)", "input_type": "select", "options": [(1, "Typical angina"), (2, "Atypical angina"), (3, "Non-anginal pain"), (4, "Asymptomatic")], "default": 3},
    {"name": "trestbps", "label": "Resting Blood Pressure (trestbps)", "input_type": "number", "min": 50, "max": 300, "step": 1, "default": 120},
    {"name": "chol", "label": "Cholesterol (chol)", "input_type": "number", "min": 50, "max": 1000, "step": 1, "default": 200},
    {"name": "fbs", "label": "Fasting Blood Sugar (fbs)", "input_type": "select", "options": [(0, "No"), (1, "Yes")], "default": 0},
    {"name": "restecg", "label": "Resting ECG (restecg)", "input_type": "select", "options": [(0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")], "default": 0},
    {"name": "thalach", "label": "Maximum Heart Rate (thalach)", "input_type": "number", "min": 50, "max": 250, "step": 1, "default": 150},
    {"name": "exang", "label": "Exercise Induced Angina (exang)", "input_type": "select", "options": [(0, "No"), (1, "Yes")], "default": 0},
    {"name": "oldpeak", "label": "Oldpeak", "input_type": "number", "min": 0, "max": 10, "step": 0.1, "default": 1.0},
    {"name": "slope", "label": "Slope", "input_type": "select", "options": [(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], "default": 2},
    {"name": "ca", "label": "Number of Major Vessels (ca)", "input_type": "number", "min": 0, "max": 4, "step": 1, "default": 0},
    {"name": "thal", "label": "Thal", "input_type": "select", "options": [(3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")], "default": 3},
]

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------
MODEL_ERROR: Optional[str] = None
MODEL = None
FEATURE_NAMES: Optional[List[str]] = None


def load_model() -> Tuple[Optional[Any], Optional[List[str]], Optional[str]]:
    """Load the trained model from disk and return the model and expected feature names."""
    if not MODEL_PATH.exists():
        return None, None, f"Model file not found at {MODEL_PATH}"

    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            model = bundle.get("model")
            feature_names = bundle.get("features")
        else:
            model = bundle
            feature_names = getattr(model, "feature_names_in_", None)

        if model is None:
            return None, None, "The model object could not be loaded."

        if feature_names is None:
            feature_names = [field["name"] for field in FIELD_DEFINITIONS]

        return model, [str(name) for name in feature_names], None
    except Exception as exc:  # pragma: no cover - defensive error handling
        return None, None, f"Failed to load model: {exc}"


MODEL, FEATURE_NAMES, MODEL_ERROR = load_model()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_default_values() -> Dict[str, Any]:
    """Return default values for the form inputs."""
    return {field["name"]: field.get("default") for field in FIELD_DEFINITIONS}


def validate_and_collect_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the form values and return a normalized dictionary of patient inputs."""
    values: Dict[str, Any] = {}

    for field in FIELD_DEFINITIONS:
        name = field["name"]
        raw_value = form_data.get(name, "")

        if raw_value in (None, ""):
            raise ValueError(f"{field['label']} is required.")

        if field["input_type"] == "select":
            valid_options = {str(option[0]) for option in field.get("options", [])}
            if str(raw_value) not in valid_options:
                raise ValueError(f"{field['label']} contains an invalid option.")
            values[name] = int(raw_value)
        else:
            try:
                numeric_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{field['label']} must be numeric.") from exc

            minimum = field.get("min")
            maximum = field.get("max")
            if minimum is not None and numeric_value < minimum:
                raise ValueError(f"{field['label']} must be at least {minimum}.")
            if maximum is not None and numeric_value > maximum:
                raise ValueError(f"{field['label']} must be at most {maximum}.")
            values[name] = numeric_value

    return values


def build_model_input(values: Dict[str, Any]) -> pd.DataFrame:
    """Convert the validated form values into a DataFrame that matches the model's expected feature order."""
    if MODEL is None:
        raise RuntimeError("Model is not available. Please check the model file.")

    if FEATURE_NAMES:
        ordered_values = []
        for feature in FEATURE_NAMES:
            if feature not in values:
                raise ValueError(f"Model expected feature '{feature}' but it was not provided.")
            ordered_values.append(values[feature])
        return pd.DataFrame([ordered_values], columns=FEATURE_NAMES)

    return pd.DataFrame([values[field["name"]] for field in FIELD_DEFINITIONS]).T


def get_prediction_result(values: Dict[str, Any]) -> Dict[str, Any]:
    """Run prediction and return the result payload for the UI."""
    if MODEL is None:
        raise RuntimeError(MODEL_ERROR or "The model could not be loaded.")

    model_input = build_model_input(values)
    prediction = MODEL.predict(model_input)[0]
    prediction_value = int(prediction)

    confidence = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(model_input)[0]
        if len(probabilities) > 1:
            confidence = float(probabilities[1])
        else:
            confidence = float(probabilities[0])

    if prediction_value == 1:
        label = "Heart Disease"
        if confidence is not None:
            if confidence >= 0.7:
                risk_level = "High Risk"
                recommendation = "Seek medical attention promptly and follow up with a cardiologist."
            else:
                risk_level = "Medium Risk"
                recommendation = "Schedule a clinical review and discuss lifestyle and monitoring steps."
        else:
            risk_level = "High Risk"
            recommendation = "Seek medical attention promptly and follow up with a cardiologist."
    else:
        label = "Healthy"
        if confidence is not None and confidence <= 0.3:
            risk_level = "Low Risk"
            recommendation = "Maintain a healthy lifestyle, stay active, and keep regular checkups."
        else:
            risk_level = "Low Risk"
            recommendation = "Maintain a healthy lifestyle and continue routine wellness checks."

    return {
        "prediction": label,
        "risk_level": risk_level,
        "confidence": confidence,
        "recommendation": recommendation,
        "prediction_code": prediction_value,
    }


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Render the main prediction page and process form submissions."""
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    if request.method == "POST":
        try:
            values = validate_and_collect_form(request.form.to_dict())
            result = get_prediction_result(values)
        except ValueError as exc:
            error_message = str(exc)
        except Exception as exc:  # pragma: no cover - defensive error handling
            error_message = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        fields=FIELD_DEFINITIONS,
        defaults=get_default_values(),
        result=result,
        error_message=error_message,
        model_error=MODEL_ERROR,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

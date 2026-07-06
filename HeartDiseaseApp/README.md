# Heart Disease Risk Assessment Agent

A professional Flask web application for predicting heart disease risk from patient clinical details using a pre-trained machine learning model.

## Features
- Modern responsive UI
- Input validation
- Model loading from heart_disease_model.pkl
- Prediction result card with confidence and recommendations
- Reset button and smooth animations

## Installation

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment
Windows:
```bash
venv\Scripts\activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

## Run the app
```bash
python app.py
```

Open your browser at:
```text
http://127.0.0.1:5000
```

## Notes
- Make sure heart_disease_model.pkl is present in the same folder as app.py.
- The app uses Flask and scikit-learn for prediction.

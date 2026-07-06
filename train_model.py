import joblib
import pandas as pd
from io import StringIO
from urllib.request import urlopen
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
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
TARGET_COLUMN = "target"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def main() -> None:
    print("Downloading heart disease dataset...")
    raw_data = urlopen(DATA_URL, timeout=60).read().decode("utf-8")
    df = pd.read_csv(StringIO(raw_data), header=None, names=[*FEATURE_COLUMNS, TARGET_COLUMN])
    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, "heart_disease_model.pkl")

    print(f"Training complete. Accuracy: {accuracy:.2f}")
    print("Model saved to heart_disease_model.pkl")
    print("Example probability:", model.predict_proba(X_test.iloc[[0]])[0][1])


if __name__ == "__main__":
    main()
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "credit_scoring_data.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "risk_flag"
MODEL_CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
}


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def build_pipeline(model_name: str):
    categorical_features = ["employment_status"]
    numeric_features = [
        col for col in load_dataset().columns if col not in categorical_features + [TARGET]
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = MODEL_CANDIDATES[model_name]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def evaluate_model(model_name: str, X_train, X_test, y_train, y_test):
    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return pipeline, metrics


def main():
    df = load_dataset()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = []
    best_model = None
    best_name = None
    best_metrics = None

    for model_name in MODEL_CANDIDATES:
        pipeline, metrics = evaluate_model(model_name, X_train, X_test, y_train, y_test)
        results.append({"model": model_name, **metrics})

        if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_model = pipeline
            best_name = model_name
            best_metrics = metrics

    print("Model comparison")
    for result in results:
        print(f"- {result['model']}: accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, recall={result['recall']:.4f}, f1={result['f1']:.4f}, roc_auc={result['roc_auc']:.4f}")

    print(f"\nBest model: {best_name}")
    print(json.dumps(best_metrics, indent=2))

    joblib.dump(best_model, MODEL_DIR / "best_credit_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "metrics": best_metrics}, file, indent=2)

    print(f"Saved model to {MODEL_DIR / 'best_credit_model.joblib'}")


if __name__ == "__main__":
    main()

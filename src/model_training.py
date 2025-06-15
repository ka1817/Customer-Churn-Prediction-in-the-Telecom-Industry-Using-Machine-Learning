import os
import joblib
import logging
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data

# Setup logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'pipeline.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Load and preprocess data
df = data_ingestion()
X, y, preprocessor = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model definitions and hyperparameter grids
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVC": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

param_grids = {
    "RandomForest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10],
        "classifier__min_samples_split": [2, 5],
    },
    "LogisticRegression": {
        "classifier__C": [0.01, 0.1, 1],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"]
    },
    "SVC": {
        "classifier__C": [0.1, 1],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"]
    },
    "DecisionTree": {
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "classifier__n_estimators": [100, 200, 250],
        "classifier__learning_rate": [0.01, 0.1, 0.03],
        "classifier__max_depth": [3, 5, 7]
    }
}

models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telecom Churn ML Models 1")

for name, model in models.items():
    logging.info(f"Training model: {name}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    grid = GridSearchCV(pipeline, param_grid=param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    # Minimal metrics logging
    logging.info(f"Model: {name} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | ROC AUC: {auc:.4f}")

    with mlflow.start_run(run_name=name):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(best_model, name)

    model_path = os.path.join(models_dir, f"{name}_best_model.pkl")
    joblib.dump(best_model, model_path)
    logging.info(f"Saved {name} to {model_path}")


# src/model_training.py

import os
import joblib
import logging
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data

# Setup logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'pipeline.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telecom Churn ML Models")

df = data_ingestion()
X, y, preprocessor = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20]
        }
    },
    "SVC": {
        "model": SVC(probability=True),
        "params": {
            "classifier__C": [1, 10],
            "classifier__kernel": ["linear", "rbf"]
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "classifier__C": [0.1, 1, 10]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 150],
            "classifier__learning_rate": [0.05, 0.1]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "classifier__max_depth": [None, 10, 20],
            "classifier__criterion": ["gini", "entropy"]
        }
    }
}

models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

for name, model_config in models.items():
    logging.info(f"Training model: {name}")
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_config["model"])
        ])

        grid = GridSearchCV(pipe, model_config["params"], cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(best_model, name)

        model_path = os.path.join(models_dir, f"{name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        logging.info(f"Saved best {name} model to {model_path}")
        logging.info(f"Metrics - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")

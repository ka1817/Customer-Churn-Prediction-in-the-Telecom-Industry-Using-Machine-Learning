import os
import joblib
import importlib.util
import warnings
warnings.filterwarnings("ignore")

def test_model_training_creates_model_files():
    # Import and execute model_training.py
    model_training_path = os.path.join(os.path.dirname(__file__), "..", "src", "model_training.py")
    spec = importlib.util.spec_from_file_location("model_training", model_training_path)
    model_training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_training)

    # Define expected model filenames
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    expected_models = [
        "RandomForest_best_model.pkl",
        "LogisticRegression_best_model.pkl",
        "SVC_best_model.pkl",
        "DecisionTree_best_model.pkl",
        "GradientBoosting_best_model.pkl"
    ]

    for model_file in expected_models:
        path = os.path.join(models_dir, model_file)
        assert os.path.isfile(path), f"Missing model file: {model_file}"
        try:
            model = joblib.load(path)
        except Exception as e:
            assert False, f"Failed to load model {model_file}: {e}"

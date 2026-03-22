from pathlib import Path

# Base project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / "customer_churn.csv"
CLUSTERED_DATA_PATH = BASE_DIR / "customer_churn_with_clusters.csv"
PREDICTION_RESULTS_PATH = BASE_DIR / "churn_prediction_results.csv"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
MODEL_COMPARISON_PATH = BASE_DIR / "model_comparison.csv"

# Model path
MODEL_PATH = BASE_DIR / "rf_churn_model.pkl"

# Random settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Clustering settings
N_CLUSTERS = 4

# Target column
TARGET_COLUMN = "Churn"

# Columns that should not be used as features
DROP_COLUMNS = ["customerID"] 
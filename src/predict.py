import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    RAW_DATA_PATH,
    PREDICTION_RESULTS_PATH,
    DROP_COLUMNS,
)
from src.data_preprocessing import clean_data, encode_target, engineer_features
from src.clustering import prepare_clustering_data, run_kmeans


def risk_level(prob):
    """
    Convert churn probability into business risk label.
    """
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def retention_action(row):
    """
    Assign retention recommendation based on risk level.
    """
    if row["RiskLevel"] == "High Risk":
        return "Offer discount or retention call"
    elif row["RiskLevel"] == "Medium Risk":
        return "Send engagement email"
    else:
        return "Maintain regular service"


def add_cluster_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster labels before prediction.
    """
    df = df.copy()

    _, X_scaled, _ = prepare_clustering_data(df)
    _, cluster_labels = run_kmeans(X_scaled)

    df["Cluster"] = cluster_labels
    return df


def load_model_artifact(path=MODEL_PATH):
    """
    Load saved model artifact.
    """
    artifact = joblib.load(path)
    return artifact


def prepare_prediction_data(df: pd.DataFrame):
    """
    Prepare input data for prediction.
    """
    df = df.copy()
    df = clean_data(df)
    df = encode_target(df)
    df = engineer_features(df)
    df = add_cluster_feature(df)

    original_df = df.copy()

    X = df.drop(columns=["Churn"], errors="ignore")
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    return original_df, X


def predict_new_data(input_path=RAW_DATA_PATH, output_path=PREDICTION_RESULTS_PATH):
    """
    Predict churn on input CSV and save results.
    """
    df = pd.read_csv(input_path)

    original_df, X = prepare_prediction_data(df)

    artifact = load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = X.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = original_df.copy()
    results["PredictedChurn"] = predictions
    results["ChurnProbability"] = probabilities
    results["RiskLevel"] = results["ChurnProbability"].apply(risk_level)
    results["RetentionAction"] = results.apply(retention_action, axis=1)

    if "PredictedChurn" in results.columns:
        results["PredictedChurn"] = results["PredictedChurn"].map({1: "Yes", 0: "No"})

    results.to_csv(output_path, index=False)
    print(f"Prediction results saved to: {output_path}")

    print("\n=== Sample Predictions ===")
    preview_cols = [col for col in [
        "customerID",
        "Cluster",
        "PredictedChurn",
        "ChurnProbability",
        "RiskLevel",
        "RetentionAction"
    ] if col in results.columns]

    print(results[preview_cols].head())

    return results


if __name__ == "__main__":
    predict_new_data()
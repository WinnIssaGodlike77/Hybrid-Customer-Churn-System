import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import MODEL_PATH, MODEL_COMPARISON_PATH, RANDOM_STATE
from src.data_preprocessing import (
    load_data,
    clean_data,
    encode_target,
    engineer_features,
    prepare_features_and_target,
    one_hot_encode_features,
    split_data,
)
from src.clustering import prepare_clustering_data, run_kmeans


def add_cluster_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cluster labels to dataframe before model training.
    """
    df = df.copy()

    _, X_scaled, _ = prepare_clustering_data(df)
    _, cluster_labels = run_kmeans(X_scaled)

    df["Cluster"] = cluster_labels
    return df


def prepare_training_data():
    """
    Full workflow for preparing training data.
    """
    df = load_data()
    df = clean_data(df)
    df = encode_target(df)
    df = engineer_features(df)
    df = add_cluster_feature(df)

    X, y = prepare_features_and_target(df)
    X = one_hot_encode_features(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return df, X, X_train, X_test, y_train, y_test


def align_test_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Ensure test columns match training columns.
    """
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_test


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    """
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr_model.fit(X_train, y_train)
    return lr_model


def evaluate_basic_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model using accuracy and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "ROC_AUC": round(roc_auc, 4)
    }


def save_model_artifact(model, feature_columns, path=MODEL_PATH):
    """
    Save model and training feature columns together.
    """
    artifact = {
        "model": model,
        "feature_columns": feature_columns
    }
    joblib.dump(artifact, path)
    print(f"Model saved to: {path}")


def save_model_comparison(results, path=MODEL_COMPARISON_PATH):
    """
    Save model comparison results to CSV.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(path, index=False)
    print(f"Model comparison saved to: {path}")


def training_workflow():
    """
    Full model training workflow.
    """
    _, X, X_train, X_test, y_train, y_test = prepare_training_data()

    X_test = align_test_columns(X_train, X_test)

    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)

    rf_result = evaluate_basic_model(rf_model, X_test, y_test, "Random Forest")
    lr_result = evaluate_basic_model(lr_model, X_test, y_test, "Logistic Regression")

    save_model_artifact(rf_model, list(X.columns))
    save_model_comparison([rf_result, lr_result])

    return rf_model, lr_model


if __name__ == "__main__":
    training_workflow()
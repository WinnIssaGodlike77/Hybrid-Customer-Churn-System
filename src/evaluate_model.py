import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import MODEL_PATH, FEATURE_IMPORTANCE_PATH
from src.train_model import prepare_training_data, align_test_columns


def load_model_artifact(path=MODEL_PATH):
    """
    Load saved model artifact.
    """
    artifact = joblib.load(path)
    return artifact


def get_feature_importance(model, feature_columns):
    """
    Extract feature importance from Random Forest model.
    """
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        return importance_df

    return pd.DataFrame(columns=["Feature", "Importance"])


def save_feature_importance(importance_df, path=FEATURE_IMPORTANCE_PATH):
    """
    Save feature importance to CSV.
    """
    importance_df.to_csv(path, index=False)
    print(f"Feature importance saved to: {path}")


def evaluate_model():
    """
    Evaluate trained model with detailed metrics.
    """
    artifact = load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    _, _, X_train, X_test, y_train, y_test = prepare_training_data()

    # Align test data to saved training feature structure
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    X_train = X_train.reindex(columns=feature_columns, fill_value=0)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("=== Model Evaluation ===")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("\n=== Confusion Matrix ===")
    print(cm)
    print("\n=== Classification Report ===")
    print(report)

    importance_df = get_feature_importance(model, feature_columns)
    if not importance_df.empty:
        save_feature_importance(importance_df)
        print("\n=== Top 10 Important Features ===")
        print(importance_df.head(10))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


if __name__ == "__main__":
    evaluate_model()
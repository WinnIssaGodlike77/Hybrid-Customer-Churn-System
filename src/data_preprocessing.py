import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, TARGET_COLUMN, DROP_COLUMNS, TEST_SIZE, RANDOM_STATE

def load_data(path=RAW_DATA_PATH):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the churn dataset.
    """
    df = df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Remove extra spaces from object/string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target column Churn from Yes/No to 1/0.
    """
    df = df.copy()

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful business-oriented features.
    """
    df = df.copy()

    if "tenure" in df.columns:
        df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)
        df["IsLoyalCustomer"] = (df["tenure"] >= 24).astype(int)

    if "MonthlyCharges" in df.columns:
        median_monthly = df["MonthlyCharges"].median()
        df["HighMonthlyCharge"] = (df["MonthlyCharges"] > median_monthly).astype(int)

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AvgChargePerMonth"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Prepare X and y for machine learning.
    """
    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    # Drop non-feature columns
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    return X, y


def one_hot_encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns into numeric columns.
    """
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def preprocess_full_data():
    """
    Full preprocessing workflow:
    load -> clean -> target encode -> feature engineering -> split
    """
    df = load_data()
    df = clean_data(df)
    df = encode_target(df)
    df = engineer_features(df)

    X, y = prepare_features_and_target(df)
    X = one_hot_encode_features(X)

    X_train, X_test, y_train, y_test = split_data(X, y)

    return df, X, y, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df, X, y, X_train, X_test, y_train, y_test = preprocess_full_data()

    print("Full dataset shape:", df.shape)
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
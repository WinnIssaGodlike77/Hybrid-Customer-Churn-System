import pandas as pd


def print_section(title: str):
    """
    Print section header in terminal.
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def save_dataframe(df: pd.DataFrame, path):
    """
    Save DataFrame to CSV file.
    """
    df.to_csv(path, index=False)
    print(f"Saved file to: {path}")


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a safe copy of dataframe.
    """
    return df.copy()
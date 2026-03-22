import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.config import N_CLUSTERS, RANDOM_STATE, CLUSTERED_DATA_PATH
from src.data_preprocessing import load_data, clean_data, encode_target, engineer_features


def prepare_clustering_data(df: pd.DataFrame):
    """
    Select and scale features for clustering.
    """
    cluster_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    X_cluster = df[cluster_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    return X_cluster, X_scaled, scaler


def run_kmeans(X_scaled, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
    """
    Fit KMeans clustering model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels


def add_cluster_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add business-friendly cluster labels.
    """
    df = df.copy()

    cluster_name_map = {
        0: "Segment 0",
        1: "Segment 1",
        2: "Segment 2",
        3: "Segment 3"
    }

    df["ClusterName"] = df["Cluster"].map(cluster_name_map)
    return df


def save_clustered_data(df: pd.DataFrame, path=CLUSTERED_DATA_PATH):
    """
    Save clustered dataset to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Clustered data saved to: {path}")


def clustering_workflow():
    """
    Full clustering workflow.
    """
    df = load_data()
    df = clean_data(df)
    df = encode_target(df)
    df = engineer_features(df)

    _, X_scaled, scaler = prepare_clustering_data(df)
    kmeans, cluster_labels = run_kmeans(X_scaled)

    df["Cluster"] = cluster_labels

    silhouette = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette:.4f}")

    df = add_cluster_names(df)
    save_clustered_data(df)

    return df, kmeans, scaler, silhouette


if __name__ == "__main__":
    df_clustered, kmeans_model, scaler_model, sil_score = clustering_workflow()
    print(df_clustered[["tenure", "MonthlyCharges", "TotalCharges", "Cluster", "ClusterName"]].head())
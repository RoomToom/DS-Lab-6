import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_style('whitegrid')


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and merging red and white wine datasets.

    """
    frames = []
    for path, wine_type in ((red_path, 'red'), (white_path, 'white')):
        frame = pd.read_csv(path, sep=';')
        frame.columns = frame.columns.str.strip()
        if frame.empty or 'quality' not in frame:
            raise ValueError('Each dataset must contain rows and a quality column.')
        if frames and set(frame.columns) != set(frames[0].columns) - {'type'}:
            raise ValueError('The wine datasets must have the same columns.')
        frame = frame.apply(pd.to_numeric, errors='raise')
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError('No complete rows remain in a wine dataset.')
        frame['type'] = wine_type
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    return data


def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from a dataset using the IQR method."""
    values = df[list(features)]
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    valid = values.ge(q1 - 1.5 * iqr) & values.le(q3 + 1.5 * iqr)
    df_clean = df.loc[valid.all(axis=1)].copy()

    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encoding of categorical variables."""
    data_encoded = pd.get_dummies(data, columns=['type'], dtype=int)

    return data_encoded


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Standardize features using statistics fitted on X_train."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_train_scaled, scaler.transform(X_test)
    return X_train_scaled


def apply_pca(X: pd.DataFrame) -> np.ndarray:
    """
    Application of the principal component analysis (PCA) method.
    """
    X_scaled = scale_features(X)

    X_pca = PCA(n_components=2, svd_solver='full').fit_transform(X_scaled)
    return X_pca


def perform_clustering(X_pca: np.ndarray) -> dict[str, float | np.ndarray]:
    """Performing clustering based on PCA results."""
    if len(X_pca) < 4 or len(np.unique(X_pca, axis=0)) < 3:
        raise ValueError('At least four samples and three distinct points are required.')
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_pca)
    results = {'labels': labels, 'silhouette_score': silhouette_score(X_pca, labels)}

    return results


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'results'
    output.mkdir(exist_ok=True)
    data = load_data(root / 'winequality-red.csv', root / 'winequality-white.csv')
    input_rows = len(data)

    # Removing outliers
    features = data.columns.drop(['quality', 'type'])
    data = remove_outliers(data, features)
    cleaned = data.copy()

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Unsupervised learning task
    X_features = data.drop(['quality'], axis=1)
    X_pca = apply_pca(X_features)
    clustering_results = perform_clustering(X_pca)
    print("\nClustering Results:")
    print(f"Silhouette coefficient: {clustering_results['silhouette_score']:.4f}")
    cleaned['PC1'] = X_pca[:, 0]
    cleaned['PC2'] = X_pca[:, 1]
    cleaned['cluster'] = clustering_results['labels']
    cleaned.to_csv(output / 'wine_clusters.csv', index=False)
    counts = cleaned['cluster'].value_counts().sort_index()
    summary = {'input_rows': input_rows, 'cleaned_rows': len(cleaned),
               'removed_rows': input_rows - len(cleaned), 'n_clusters': 3,
               'silhouette_score': clustering_results['silhouette_score'],
               'cluster_sizes': {str(k): int(v) for k, v in counts.items()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    # Visualization of clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    hue=clustering_results['labels'],
                    palette='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(output / 'clusters.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


def kmeans_and_score(encoder, n):
    """
    Silhouette Score: [-1, 1]
    Higher is better
    Measures how similar a point is to its own cluster vs other clusters
    """
    k_values = range(2, 11)
    scores = []

    best_score = -float("inf")
    best_k = None
    best_labels = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=n, random_state=42)
        labels = kmeans.fit_predict(encoder.norm_matrix)

        score = silhouette_score(encoder.norm_matrix, labels, metric='cosine')
        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return {
        "k_values": list(k_values),
        "scores": scores,
        "best_k": best_k,
        "best_score": best_score,
        "best_labels": best_labels,
        "encoder": encoder
    }



def kmeans_and_score_davies_bouldin(encoder, n):
    """
    Davies-Bouldin Score: [0, inf]
    Lower is better (opposite of others!)
    Average similarity ratio of each cluster with its most similar cluster
    """
    k_values = range(2, 11)
    scores = []
    best_score = float("inf")  # Note: lower is better!
    best_k = None
    best_labels = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=n, random_state=42)
        labels = kmeans.fit_predict(encoder.norm_matrix)

        # Convert sparse to dense if needed
        import scipy.sparse as sp
        matrix = encoder.norm_matrix.toarray() if sp.issparse(encoder.norm_matrix) else encoder.norm_matrix

        score = davies_bouldin_score(matrix, labels)
        scores.append(score)

        if score < best_score:  # Note: lower is better!
            best_score = score
            best_k = k
            best_labels = labels

    return {
        "k_values": list(k_values),
        "scores": scores,
        "best_k": best_k,
        "best_score": best_score,
        "best_labels": best_labels,
        "encoder": encoder,
        "metric": "davies_bouldin"
    }


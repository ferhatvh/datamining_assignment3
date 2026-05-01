from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import scipy.sparse as sp
import numpy as np


def cluster_and_score(encoder, method='kmeans', n_init=20, **kwargs):

    k_values = range(2, 11)
    scores = []
    best_score = -float("inf")
    best_k = None
    best_labels = None
    best_model = None

    matrix = encoder.norm_matrix
    if method == 'hierarchical' and sp.issparse(matrix):
        matrix = matrix.toarray()
    
    for k in k_values:
        if method == 'kmeans' or isinstance(method, type(KMeans)):
            clusterer = KMeans(n_clusters=k, n_init=n_init, random_state=42, **kwargs)
        elif method == 'hierarchical':
            linkage = kwargs.get('linkage', 'ward')
            clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        elif method == 'spectral':
            affinity = kwargs.get('affinity', 'rbf')
            clusterer = SpectralClustering(
                n_clusters=k, 
                n_init=n_init,
                affinity=affinity,
                assign_labels='kmeans'
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        labels = clusterer.fit_predict(matrix)
        metric = 'cosine' if sp.issparse(encoder.norm_matrix) else 'euclidean'
        score = silhouette_score(encoder.norm_matrix, labels, metric=metric)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = clusterer
    
    return {
        "k_values": list(k_values),
        "scores": scores,
        "best_k": best_k,
        "best_score": best_score,
        "best_labels": best_labels,
        "best_model": best_model,
        "encoder": encoder
    }


def get_top_terms_per_cluster(result, n_terms=10):
    encoder = result["encoder"]
    labels = result["best_labels"]
    k = result["best_k"]
    
    if encoder.terms is None:
        return None
    
    cluster_terms = {}
    
    matrix = encoder.dens_matrix
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    
    for cluster_id in range(k):
        cluster_docs = matrix[labels == cluster_id]
        mean_weights = cluster_docs.mean(axis=0)
        top_indices = mean_weights.argsort()[-n_terms:][::-1]
        top_terms = [(encoder.terms[i], mean_weights[i]) for i in top_indices]
        
        cluster_terms[cluster_id] = top_terms
    
    return cluster_terms


def print_cluster_summary(result, n_terms=10):
    """Print a summary of clusters with top terms."""
    print(f"\n{'='*70}")
    print(f"Clustering Summary")
    print('='*70)
    print(f"Best k: {result['best_k']}")
    print(f"Best silhouette score: {result['best_score']:.4f}")
    print(f"\nCluster sizes:")
    
    labels = result["best_labels"]
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} documents")
    
    cluster_terms = get_top_terms_per_cluster(result, n_terms)
    
    if cluster_terms:
        print(f"\nTop {n_terms} terms per cluster:")
        for cluster_id, terms in cluster_terms.items():
            print(f"\nCluster {cluster_id}:")
            term_strs = [f"{term} ({weight:.3f})" for term, weight in terms]
            print(f"  {', '.join(term_strs)}")

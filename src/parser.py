import pandas as pd
import matplotlib.pyplot as plt


from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp


def readCSV(path="data/articles.csv"):
    df = pd.read_csv(path)
    return df


def plot_results(results_dict, title):

    plt.figure(figsize=(10,6))

    for name, res in results_dict.items():
        plt.plot(res["k_values"], res["scores"], marker='o', label=name)

    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.legend()
    plt.show()


def visualize_best_clusters_mds(results_dict, title, max_docs=400):
    """
    Visualize clustering results using sampled MDS projection.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results
    title : str
        Plot title
    max_docs : int
        Maximum number of sampled documents for MDS
    """

    n_methods = len(results_dict)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(6 * n_methods, 5)
    )

    if n_methods == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(results_dict.items()):

        encoder = result["encoder"]
        labels = result["best_labels"]
        k = result["best_k"]
        score = result["best_score"]

        if sp.issparse(encoder.norm_matrix):
            matrix_dense = encoder.norm_matrix.toarray()
        else:
            matrix_dense = encoder.norm_matrix
        if matrix_dense.shape[0] > max_docs:

            sample_indices = []

            docs_per_cluster = max_docs // k

            for cluster_id in range(k):

                cluster_docs = np.where(labels == cluster_id)[0]

                n_sample = min(
                    docs_per_cluster,
                    len(cluster_docs)
                )

                sampled = np.random.choice(
                    cluster_docs,
                    n_sample,
                    replace=False
                )

                sample_indices.extend(sampled)

            sample_indices = np.array(sample_indices)

            matrix_sample = matrix_dense[sample_indices]
            labels_sample = labels[sample_indices]

        else:
            matrix_sample = matrix_dense
            labels_sample = labels

        print(f"Running MDS for {name} on {len(labels_sample)} documents...")

        mds = MDS(
            n_components=2,
            random_state=42,
            dissimilarity='euclidean',
            n_jobs=-1
        )

        coords_2d = mds.fit_transform(matrix_sample)

        scatter = axes[idx].scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=labels_sample,
            cmap='tab10',
            alpha=0.6,
            s=30
        )

        for cluster_id in range(k):

            cluster_points = coords_2d[
                labels_sample == cluster_id
            ]

            if len(cluster_points) == 0:
                continue

            center = cluster_points.mean(axis=0)

            axes[idx].scatter(
                center[0],
                center[1],
                c='red',
                marker='X',
                s=200,
                edgecolors='black',
                linewidths=2,
                label='Center' if cluster_id == 0 else ''
            )

        axes[idx].set_title(
            f"{name}\nk={k}, Score={score:.3f}",
            fontsize=12,
            fontweight='bold'
        )

        axes[idx].set_xlabel("MDS Dimension 1")
        axes[idx].set_ylabel("MDS Dimension 2")

        axes[idx].grid(True, alpha=0.3)

        if idx == 0:
            axes[idx].legend()

        plt.colorbar(
            scatter,
            ax=axes[idx],
            label='Cluster'
        )

    plt.suptitle(
        title,
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.show()


def visualize_similarity_heatmap(results_dict, title, max_docs=100):
    """
    Visualize cosine similarity matrices as heatmaps.
    """

    n_methods = len(results_dict)

    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(6 * n_methods, 5)
    )

    if n_methods == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(results_dict.items()):

        encoder = result["encoder"]
        labels = result["best_labels"]
        k = result["best_k"]
        score = result["best_score"]

        if sp.issparse(encoder.norm_matrix):
            matrix_dense = encoder.norm_matrix.toarray()
        else:
            matrix_dense = encoder.norm_matrix

        if matrix_dense.shape[0] > max_docs:

            sample_indices = []

            docs_per_cluster = max_docs // k

            for cluster_id in range(k):

                cluster_docs = np.where(labels == cluster_id)[0]

                n_sample = min(
                    docs_per_cluster,
                    len(cluster_docs)
                )

                sampled = np.random.choice(
                    cluster_docs,
                    n_sample,
                    replace=False
                )

                sample_indices.extend(sampled)

            sample_indices = np.array(sample_indices)

            matrix_sample = matrix_dense[sample_indices]
            labels_sample = labels[sample_indices]

        else:
            matrix_sample = matrix_dense
            labels_sample = labels

        similarity_matrix = cosine_similarity(matrix_sample)
        sort_idx = np.argsort(labels_sample)

        similarity_sorted = similarity_matrix[
            sort_idx
        ][:, sort_idx]

        labels_sorted = labels_sample[sort_idx]

        im = axes[idx].imshow(
            similarity_sorted,
            cmap='viridis',
            aspect='auto',
            interpolation='nearest'
        )

        cluster_boundaries = []

        current_cluster = labels_sorted[0]

        for i, label in enumerate(labels_sorted):

            if label != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = label

        for boundary in cluster_boundaries:

            axes[idx].axhline(
                boundary - 0.5,
                color='red',
                linewidth=2,
                alpha=0.7
            )

            axes[idx].axvline(
                boundary - 0.5,
                color='red',
                linewidth=2,
                alpha=0.7
            )

        axes[idx].set_title(
            f"{name}\nk={k}, Score={score:.3f}",
            fontsize=12,
            fontweight='bold'
        )

        axes[idx].set_xlabel(
            "Documents (sorted by cluster)"
        )

        axes[idx].set_ylabel(
            "Documents (sorted by cluster)"
        )

        plt.colorbar(
            im,
            ax=axes[idx],
            label='Cosine Similarity'
        )

    plt.suptitle(
        title,
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.show()


def visualize_best_clusters(
    results_dict,
    title,
    method='mds'
):
    """
    Wrapper visualization function.
    """

    if method == 'mds':

        visualize_best_clusters_mds(
            results_dict,
            title
        )

    elif method == 'heatmap':

        visualize_similarity_heatmap(
            results_dict,
            title
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )
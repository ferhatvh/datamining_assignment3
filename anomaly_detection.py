from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
import numpy as np


def detect_anomalies(encoder, method='lof', n_anomalies=50, **kwargs):
    matrix = encoder.norm_matrix
    
    if method == 'lof':
        n_neighbors = kwargs.get('n_neighbors', 20)
        metric = kwargs.get('metric', 'cosine')
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=metric,
            contamination='auto',
            novelty=False
        )
        
        predictions = lof.fit_predict(matrix)
        scores = -lof.negative_outlier_factor_
        
    elif method == 'isolation_forest':
        contamination = kwargs.get('contamination', n_anomalies / matrix.shape[0])
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(matrix)
        scores = -iso_forest.score_samples(matrix)
        
    elif method == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 20)
        metric = kwargs.get('metric', 'cosine')
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nbrs.fit(matrix)
        
        distances, indices = nbrs.kneighbors(matrix)
        scores = distances.mean(axis=1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    anomaly_indices = np.argsort(scores)[-n_anomalies:][::-1]
    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_scores": scores,
        "method": method
    }


def print_anomaly_summary(result, df_articles=None):
    """Print summary of detected anomalies."""
    print(f"\n{'='*70}")
    print(f"Anomaly Detection Summary ({result['method'].upper()})")
    print('='*70)
    print(f"Number of anomalies detected: {len(result['anomaly_indices'])}")
    
    anomaly_indices = result['anomaly_indices']
    scores = result['anomaly_scores']
    
    print(f"\nTop 10 most anomalous documents:")
    for i, idx in enumerate(anomaly_indices[:10]):
        score = scores[idx]
        doc_id = df_articles.iloc[idx]['doc_id'] if df_articles is not None else f"Doc_{idx}"
        print(f"  {i+1}. {doc_id} (score: {score:.4f})")
    
    print(f"\nAnomaly score statistics:")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Anomaly threshold: {scores[anomaly_indices[-1]]:.4f}")


def save_anomalies_csv(result, df_articles, output_path='anomalies.csv'):
    """Save detected anomalies to CSV file."""
    import pandas as pd
    
    anomaly_indices = result['anomaly_indices']
    anomaly_doc_ids = df_articles.iloc[anomaly_indices]['doc_id'].values
    
    anomalies_df = pd.DataFrame({'doc_id': anomaly_doc_ids})
    
    anomalies_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(anomaly_doc_ids)} anomalies to {output_path}")

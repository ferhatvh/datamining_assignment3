# Data Mining Assignment 3 – Text Clustering & Anomaly Detection

## Overview
This project applies text clustering and anomaly detection to a dataset of articles using different text representations (Bag-of-Words, TF-IDF, Word2Vec) and clustering methods (K-Means, Hierarchical, Spectral).

## Main File
All experiments, testing, and results are done in:
`clustering_tests.ipynb`

This notebook includes preprocessing, feature extraction, clustering, evaluation, anomaly detection, and visualizations.

## Project Structure
- `src/` → encoders, clustering, and utilities  
- `anomaly_detection.py` → anomaly detection methods (LOF, Isolation Forest)  
- `data/` → dataset files (`articles.csv`, `clusters.csv`, `anomalies.csv`)  
- `nltk_data/` → required NLTK resources (included locally)  
- `requirements.txt` → dependencies  

## Installation
```bash
pip install -r requirements.txt


NLTK data is included in ./nltk_data.
If issues occur, uncomment and run:

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

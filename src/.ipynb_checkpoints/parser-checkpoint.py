import pandas as pd
import matplotlib.pyplot as plt


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

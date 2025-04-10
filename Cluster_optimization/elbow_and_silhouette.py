import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def extract_embeddings(df):
    embeddings = []
    for emb in df['embedding']:
        if isinstance(emb, str):
            emb = emb.strip('[]')
            emb_list = [float(x.strip()) for x in emb.split(',')]
            embeddings.append(emb_list)
        else:
            embeddings.append(emb)
    embeddings_array = np.vstack(embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    return embeddings_scaled

def plot_elbow(inertias, K_range):
    plt.figure(figsize=(8, 5))
    scaled_inertias = [i / 1e7 for i in inertias]

    plt.plot(K_range, scaled_inertias, marker='o', color='blue')
    plt.xlabel('Number of Clusters (k)', fontsize=24)
    plt.ylabel('Inertia ($\\times 10^7$)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y') 
    plt.tight_layout()
    plt.savefig('elbow.pdf')
    plt.show()


def plot_silhouette(silhouettes, K_range):
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouettes, marker='o', color='red')
    # plt.title('Silhouette Analysis', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=24)
    plt.ylabel('Silhouette Score', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('silhouette.pdf')
    plt.show()

def main():
    df = pd.read_csv('log_data_ten_thousand.csv')
    embeddings_scaled = extract_embeddings(df)

    inertias = []
    silhouettes = []
    K_range = range(2, 31)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(embeddings_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(embeddings_scaled, labels))

    plot_elbow(inertias, K_range)
    plot_silhouette(silhouettes, K_range)

if __name__ == "__main__":
    main()

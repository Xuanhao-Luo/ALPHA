import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

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

def get_cluster_labels(kmeans, df, embeddings_scaled):
    centers = kmeans.cluster_centers_
    cluster_labels = {}

    for cluster_id in range(len(centers)):
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_points = df[cluster_mask]
        cluster_embeddings = embeddings_scaled[cluster_mask]

        if len(cluster_points) == 0:
            continue

        cluster_distances = np.linalg.norm(
            kmeans.transform(cluster_embeddings)[:, cluster_id].reshape(-1, 1), axis=1
        )
        center_point_idx = np.argmin(cluster_distances)
        center_point_label = cluster_points.iloc[center_point_idx]['label']
        cluster_labels[cluster_id] = center_point_label != '-'
    
    return cluster_labels

def evaluate_accuracy_coverage(distances, predictions, true_labels, epsilons):
    acc_list, cov_list = [], []
    true_anomalous = np.array([label != '-' for label in true_labels])

    for epsilon in epsilons:
        adjusted_pred = predictions.copy()
        adjusted_pred[distances > epsilon] = ~adjusted_pred[distances > epsilon]
        accuracy = np.mean(adjusted_pred == true_anomalous)
        coverage = np.mean(distances <= epsilon) * 100  
        acc_list.append(accuracy)
        cov_list.append(coverage)

    return acc_list, cov_list



def plot_accuracy_coverage(epsilons, accs, covs, best_eps, best_acc):
    fig, ax1 = plt.subplots(figsize=(8, 5))


    ax1.plot(epsilons, covs, label='Coverage %', color='green', linewidth=2.5)
    ax1.set_xlabel('Epsilon (Distance Threshold)', fontsize=28)
    ax1.set_ylabel('Coverage (%)',  fontsize=28)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_ylim(0, 105)  

    ax2 = ax1.twinx()
    ax2.plot(epsilons, accs, label='Accuracy',  linewidth=2.5)
    ax2.set_ylabel('Accuracy',  fontsize=28)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_ylim(0, 2) 
    ax2.set_yticks(np.linspace(0, 1, 2)) 

    ax1.axvline(x=best_eps, color='red', linestyle='--', linewidth=2.5,
                label=f'Best ε = {best_eps:.3f}\nAcc = {best_acc:.3f}\nCov = {covs[accs.index(best_acc)]:.1f}%')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=18, loc='lower right')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('epsilon.pdf')
    plt.show()


def main():
    df = pd.read_csv("log_data_ten_thousand.csv")
    embeddings_scaled = extract_embeddings(df)

    kmeans = KMeans(n_clusters=15, random_state=42, n_init=20)
    kmeans.fit(embeddings_scaled)

    cluster_labels = get_cluster_labels(kmeans, df, embeddings_scaled)
    distances = np.min(kmeans.transform(embeddings_scaled), axis=1)
    predictions = np.array([cluster_labels[c] for c in kmeans.labels_])
    true_labels = df['label'].values

    epsilons = np.linspace(np.min(distances), 100, 100)
    accs, covs = evaluate_accuracy_coverage(distances, predictions, true_labels, epsilons)

    best_idx = np.argmax(accs)
    best_eps = epsilons[best_idx]
    best_acc = accs[best_idx]

    plot_accuracy_coverage(epsilons, accs, covs, best_eps, best_acc)

if __name__ == "__main__":
    main()

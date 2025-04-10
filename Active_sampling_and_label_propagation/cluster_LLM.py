
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import os
from prompting_cluster import analyze_log_entry 

def extract_embeddings(df):
    """Extract numerical embeddings from the dataset"""
    try:
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
        return embeddings_scaled, embeddings_array, scaler

    except Exception as e:
        logging.error(f"Error processing embeddings: {str(e)}")
        return None, None, None

def get_cluster_labels_with_llm(kmeans, df, embeddings_scaled):

    cluster_labels = {}
    cluster_samples = {}

    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = kmeans.labels_ == cluster_id
        cluster_points = df[cluster_mask]
        cluster_embeddings = embeddings_scaled[cluster_mask]

        if len(cluster_points) == 0:
            continue

        cluster_distances = np.linalg.norm(
            kmeans.transform(cluster_embeddings)[:, cluster_id].reshape(-1, 1), axis=1
        )

        top_5_idx = np.argsort(cluster_distances)[:5]
        selected_samples = cluster_points.iloc[top_5_idx]

        llm_labels = []
        misclassified_normal_as_anomaly = 0  
        misclassified_anomaly_as_normal = 0  

        for _, row in selected_samples.iterrows():
            log_entry = row['remaining_log']
            true_label = 1 if row['label'] != '-' else 0 
            _, llm_decision, _ = analyze_log_entry(log_entry)  
            llm_decision = int(llm_decision)
            llm_labels.append(llm_decision)


            if true_label == 0 and llm_decision == 1:
                misclassified_normal_as_anomaly += 1
            elif true_label == 1 and llm_decision == 0:
                misclassified_anomaly_as_normal += 1


        final_cluster_label = 1 if sum(llm_labels) > 2 else 0  

        cluster_samples[cluster_id] = {
            'samples': selected_samples,
            'llm_labels': llm_labels,
            'final_label': final_cluster_label,
            'misclassified_normal_as_anomaly': misclassified_normal_as_anomaly,
            'misclassified_anomaly_as_normal': misclassified_anomaly_as_normal
        }

        print(f"\nCluster {cluster_id}:")
        print(f"  - LLM Decisions: {llm_labels}")
        print(f"  - True Normal Misclassified as Anomaly: {misclassified_normal_as_anomaly}")
        print(f"  - True Anomaly Misclassified as Normal: {misclassified_anomaly_as_normal}")
        print(f"  - Majority Vote Result: {final_cluster_label}")

        cluster_labels[cluster_id] = final_cluster_label

    return cluster_labels, cluster_samples

def assign_labels_to_sampled_data(df_sampled, cluster_labels, kmeans):

    df_sampled = df_sampled.copy()
    df_sampled['cluster'] = kmeans.labels_  
    df_sampled['expanded_label'] = df_sampled['cluster'].map(cluster_labels) 
    df_sampled['expanded_label'] = df_sampled['expanded_label'].astype(int)  

    df_sampled['true_label'] = (df_sampled['label'] != '-').astype(int)

    num_misclassified = np.sum(df_sampled['expanded_label'] != df_sampled['true_label'])
    total_samples = len(df_sampled)

    print(f"\nTotal sampled data: {total_samples}")
    print(f"Misclassified samples: {num_misclassified}")
    print(f"Misclassification rate: {num_misclassified / total_samples:.4f}")

    return df_sampled, num_misclassified

def process_data(input_file, sample_size=500, output_prefix="clustered_data_LLM"):
    """Main function to process data"""
    df_full = pd.read_csv(input_file).sample(frac=1, random_state=42).reset_index(drop=True)
    

    df_sampled = df_full.head(sample_size)

    embeddings_scaled, embeddings_orig, scaler = extract_embeddings(df_sampled)
    if embeddings_scaled is None:
        return None

    n_clusters = 15 
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(embeddings_scaled)

    cluster_labels, cluster_samples = get_cluster_labels_with_llm(kmeans, df_sampled, embeddings_scaled)

    df_labeled, misclassified_count = assign_labels_to_sampled_data(df_sampled, cluster_labels, kmeans)

    output_folder = "clustered_data"
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{output_prefix}_{sample_size}.csv")
    df_labeled.to_csv(output_file, index=False)
    
    print(f"\nProcessed dataset saved to {output_file}")
    print(f"Total Misclassified Samples: {misclassified_count}")

    return output_file, misclassified_count

if __name__ == "__main__":
    input_file = "log_data_ten_thousand.csv"
    for sample_size in range(50, 5500, 50): 
        output_file, misclassified_count = process_data(input_file, sample_size)

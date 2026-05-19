import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import random

# Load dataset
spotify_df = pd.read_csv("SpotifyData_by artists.csv")

# Select and scale only acousticness and danceability
X = spotify_df[['acousticness', 'danceability']].dropna().values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

optimal_k = 3  # you can adjust if needed

# ---------------- CLARA ---------------- #
def k_medoids_manual(X, k, max_iter=100):
    np.random.seed(42)
    m = X.shape[0]
    medoid_idxs = np.random.choice(m, k, replace=False)
    medoids = X[medoid_idxs]

    for _ in range(max_iter):
        distances = pairwise_distances(X, medoids)
        labels = np.argmin(distances, axis=1)

        new_medoids = np.array([
            X[labels == i][np.argmin(np.sum(pairwise_distances(X[labels == i], [m]), axis=1))]
            if len(X[labels == i]) > 0 else medoids[i]
            for i, m in enumerate(medoids)
        ])

        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return labels, medoids

def clara(X, k, sample_size=100, num_samples=5):
    best_labels, best_medoids = None, None
    best_cost = float('inf')
    for _ in range(num_samples):
        sample_idxs = random.sample(range(X.shape[0]), sample_size)
        X_sample = X[sample_idxs]
        labels_sample, medoids_sample = k_medoids_manual(X_sample, k)
        distances = pairwise_distances(X, medoids_sample)
        total_cost = np.sum(np.min(distances, axis=1))
        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = np.argmin(distances, axis=1)
            best_medoids = medoids_sample
    return best_labels, best_medoids

clara_labels, _ = clara(X_scaled, optimal_k)
spotify_df['CLARA_Cluster'] = clara_labels

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clara_labels, cmap='coolwarm', alpha=0.7)
plt.title("CLARA Clustering")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ---------------- CLARAN ---------------- #
def claran(X, k, sample_size=100, num_samples=5, replace_fraction=0.2):
    best_labels, best_medoids = None, None
    best_cost = float('inf')
    for _ in range(num_samples):
        sample_idxs = random.sample(range(X.shape[0]), sample_size)
        X_sample = X[sample_idxs]
        labels_sample, medoids_sample = k_medoids_manual(X_sample, k)

        num_replace = int(k * replace_fraction)
        replace_idxs = random.sample(range(k), num_replace)
        new_candidates = X[random.sample(range(X.shape[0]), num_replace)]
        for i, idx in enumerate(replace_idxs):
            medoids_sample[idx] = new_candidates[i]

        distances = pairwise_distances(X, medoids_sample)
        cost = np.sum(np.min(distances, axis=1))

        if cost < best_cost:
            best_cost = cost
            best_labels = np.argmin(distances, axis=1)
            best_medoids = medoids_sample
    return best_labels, best_medoids

claran_labels, _ = claran(X_scaled, optimal_k)
spotify_df['CLARAN_Cluster'] = claran_labels

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=claran_labels, cmap='viridis', alpha=0.7)
plt.title("CLARAN Clustering")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ----------------- Divisive -----------------

# Sample to avoid memory error (e.g., 5000 samples)
sample_size = 5000
if X_scaled.shape[0] > sample_size:
    sample_indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
    X_div_sample = X_scaled[sample_indices]
else:
    X_div_sample = X_scaled

# Perform divisive clustering on sampled data
div_linkage = linkage(X_div_sample, method='ward')
divisive_labels = fcluster(div_linkage, t=optimal_k, criterion='maxclust')

# Assign sample cluster labels back into a full array
full_divisive_labels = np.full(X_scaled.shape[0], -1)
full_divisive_labels[sample_indices] = divisive_labels
spotify_df['Divisive_Cluster'] = full_divisive_labels

# Dendrogram (only for sample)
plt.figure(figsize=(12, 5))
dendrogram(div_linkage)
plt.title("Divisive Hierarchical Clustering Dendrogram (Sampled)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Scatter plot using sample's labels
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=full_divisive_labels, cmap='plasma', alpha=0.7)
plt.title("Divisive Clustering (Sampled)")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# ---------------- Metrics ---------------- #
def compute_metrics(X, labels, method):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"[{method}] Only one cluster found, skipping metrics.\n")
        return

    inertia = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        center = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - center) ** 2)

    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    print(f"{method} Clustering Metrics:")
    print(f"Inertia: {inertia:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies:.3f}")
    print("-" * 40)

compute_metrics(X_scaled, clara_labels, "CLARA")
compute_metrics(X_scaled, claran_labels, "CLARAN")
compute_metrics(X_div_sample, divisive_labels, "Divisive")


# clustering_spotify.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from minisom import MiniSom
import random

# ------------------ Load and Prepare Data ------------------
file_path = 'SpotifyData_by artists.csv'
df = pd.read_csv(file_path)

# Use only two selected numeric columns
df = df[['acousticness', 'danceability']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Optional: Downsample for memory-intensive methods
sample_df = df.sample(n=2000, random_state=42)
X_sampled = scaler.fit_transform(sample_df)

# Helper to compute metrics
results = {}
def compute_metrics(X, labels, method):
    labels = np.array(labels)
    if len(np.unique(labels)) <= 1:
        print(f"{method}: Not enough clusters to compute metrics.")
        return
    inertia = 0
    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        center = cluster_points.mean(axis=0)
        inertia += ((cluster_points - center) ** 2).sum()

    print(f"{method}:")
    print(f"Inertia: {inertia:.3f}")
    print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, labels):.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.3f}")
    print("-" * 40)
# ------------------ Elbow Method for Optimal K ------------------
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method to Determine Optimal k for Spotify Dataset')
plt.grid(True)
plt.show()

# ------------------ KMeans ------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)
df['KMeans'] = labels_kmeans
compute_metrics(X_scaled, labels_kmeans, 'KMeans')

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.7)
plt.title("KMeans Clustering")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ------------------ KMedoids ------------------
kmedoids = KMedoids(n_clusters=3, random_state=42)
labels_kmedoids = kmedoids.fit_predict(X_sampled)
sample_df['KMedoids'] = labels_kmedoids
compute_metrics(X_sampled, labels_kmedoids, 'KMedoids (Sampled)')

plt.figure(figsize=(8,6))
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=labels_kmedoids, cmap='coolwarm', alpha=0.7)
plt.title("KMedoids Clustering (Sampled)")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ------------------ DBSCAN ------------------
dbscan = DBSCAN(eps=0.3, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_scaled)
df['DBSCAN'] = labels_dbscan
compute_metrics(X_scaled, labels_dbscan, 'DBSCAN')

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_dbscan, cmap='Spectral', alpha=0.7)
plt.title("DBSCAN Clustering")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ------------------ Agglomerative Clustering ------------------
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_sampled)
sample_df['Agglomerative'] = labels_agg
compute_metrics(X_sampled, labels_agg, 'Agglomerative (Sampled)')

plt.figure(figsize=(8,6))
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=labels_agg, cmap='plasma', alpha=0.7)
plt.title("Agglomerative Clustering (Sampled)")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ------------------ SOM ------------------
som_grid = (3, 3)
som = MiniSom(x=som_grid[0], y=som_grid[1], input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_sampled)
som.train_random(X_sampled, num_iteration=500)
bmu_indexes = np.array([som.winner(x) for x in X_sampled])
labels_som = [i * som_grid[1] + j for i, j in bmu_indexes]
sample_df['SOM'] = labels_som
compute_metrics(X_sampled, labels_som, 'SOM (Sampled)')

plt.figure(figsize=(8,6))
plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=labels_som, cmap='cubehelix', alpha=0.7)
plt.title("SOM Clustering (Sampled)")
plt.xlabel("Acousticness")
plt.ylabel("Danceability")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# ------------------ Done ------------------
print("Clustering analysis on Spotify data completed.")

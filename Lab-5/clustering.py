#Clustering Algorithms: Iris Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from minisom import MiniSom
# Load and inspect the dataset
import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent / "iris_data.csv"
df = pd.read_csv(file_path)


# Display basic information about the dataset
df.info(), df.head()

#------------------Add Header--------------------------------------------------------------------------
# load the dataset with a header
iris_df = pd.read_csv(file_path, header=None)

# Assign proper column names based on the standard Iris dataset
iris_df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Display the first few rows again
iris_df.head()

#-------------Using KMeans clustering method-----------------------------------------------------------

from sklearn.cluster import KMeans

# Extract feature columns for clustering (excluding the species column)
iris_features = iris_df.drop(columns=['species'])

# Normalize the data for better clustering
scaler = StandardScaler()
scaled_iris = scaler.fit_transform(iris_features)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_iris)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method to Determine Optimal k for Iris Dataset')
plt.show()


#--------Plot optimal number of features---------------------------------------------------------------

# Apply K-means clustering with the optimal number of clusters (k=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
iris_df['Cluster'] = kmeans.fit_predict(scaled_iris)

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['Cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('K-means Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#--------------------------New cluster table-----------------------------------------------------------

# Compare clustering results with actual species labels
comparison_df = iris_df[['species', 'Cluster']]

# Show the clustered data with actual species labels
print ("Clustered Iris Data Comparison")
print (comparison_df)


#----------Using K-Medoids (PAM) method----------------------------------------------------------------

# Implementing K-Medoids clustering manually using a robust method
def k_medoids_manual(X, k, max_iter=100):
    np.random.seed(42)
    m = X.shape[0]

    # Initialize medoids randomly
    medoid_idxs = np.random.choice(m, k, replace=False)
    medoids = X[medoid_idxs]

    for _ in range(max_iter):
        # Compute distance matrix
        distances = pairwise_distances(X, medoids, metric='euclidean')

        # Assign each point to the nearest medoid
        labels = np.argmin(distances, axis=1)

        # Update medoids by selecting the most central point in each cluster
        new_medoids = np.array([X[labels == i][np.argmin(np.sum(pairwise_distances(X[labels == i], [m], metric='euclidean'), axis=1))]
                                if len(X[labels == i]) > 0 else medoids[i]  # Keep the old medoid if cluster is empty
                                for i, m in enumerate(medoids)])

        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    return labels, medoids

# Apply the manual K-Medoids clustering
pam_labels, _ = k_medoids_manual(scaled_iris, optimal_k)

# Add clustering results to the dataframe
iris_df['PAM_Cluster'] = pam_labels

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['PAM_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('PAM (K-Medoids) Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#-------------------CLARA Method (Clustering Large Applications)---------------------------------------

import random

def clara(X, k, sample_size=40, num_samples=5):
    np.random.seed(42)
    best_labels = None
    best_medoids = None
    best_cost = float('inf')

    for _ in range(num_samples):
        # Randomly sample a subset of data
        sample_idxs = random.sample(range(X.shape[0]), sample_size)
        X_sample = X[sample_idxs]

        # Apply K-Medoids to the subset
        labels_sample, medoids_sample = k_medoids_manual(X_sample, k)

        # Compute cost (sum of distances to medoids for all points in the dataset)
        medoid_distances = pairwise_distances(X, medoids_sample, metric='euclidean')
        total_cost = np.sum(np.min(medoid_distances, axis=1))

        # Keep the best medoids based on the lowest cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = np.argmin(medoid_distances, axis=1)
            best_medoids = medoids_sample

    return best_labels, best_medoids

# Apply CLARA clustering
clara_labels, _ = clara(scaled_iris, optimal_k)

# Add clustering results to the dataframe
iris_df['CLARA_Cluster'] = clara_labels

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['CLARA_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('CLARA Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()



#-----------------------CLARAN (Clustering Large Applications based on Randomized Search)--------------

def claran(X, k, sample_size=40, num_samples=5, replace_fraction=0.2):
    np.random.seed(42)
    best_labels = None
    best_medoids = None
    best_cost = float('inf')

    for _ in range(num_samples):
        # Randomly sample a subset of data
        sample_idxs = random.sample(range(X.shape[0]), sample_size)
        X_sample = X[sample_idxs]

        # Apply K-Medoids to the subset
        labels_sample, medoids_sample = k_medoids_manual(X_sample, k)

        # Introduce random replacements in medoids (CLARAN improvement)
        num_replacements = int(k * replace_fraction)  # Replace a fraction of medoids
        replacement_idxs = random.sample(range(len(medoids_sample)), num_replacements)

        # Select new potential medoids randomly from the full dataset
        new_medoid_candidates = X[random.sample(range(X.shape[0]), num_replacements)]

        # Replace some medoids
        for i, idx in enumerate(replacement_idxs):
            medoids_sample[idx] = new_medoid_candidates[i]

        # Compute cost (sum of distances to medoids for all points in the dataset)
        medoid_distances = pairwise_distances(X, medoids_sample, metric='euclidean')
        total_cost = np.sum(np.min(medoid_distances, axis=1))

        # Keep the best clustering based on the lowest cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_labels = np.argmin(medoid_distances, axis=1)
            best_medoids = medoids_sample

    return best_labels, best_medoids

# Apply CLARAN clustering
claran_labels, _ = claran(scaled_iris, optimal_k)

# Add clustering results to the dataframe
iris_df['CLARAN_Cluster'] = claran_labels

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['CLARAN_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('CLARAN Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#------------------Using Hierarchical Clustering Algorithms: Agglomerative Clustering------------------

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Perform hierarchical clustering using Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
iris_df['Agglomerative_Cluster'] = agg_clustering.fit_predict(scaled_iris)

# Plot the dendrogram
plt.figure(figsize=(12, 6))
linkage_matrix = linkage(scaled_iris, method='ward')
dendrogram(linkage_matrix)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Agglomerative)')
plt.show()

# Scatter plot visualization (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['Agglomerative_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Agglomerative Hierarchical Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#---------------------------Hierarchical Clustering Algorithms: Divisive Clustering--------------------

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

# Perform hierarchical clustering using Divisive method (DIANA equivalent)
# Since scipy does not provide a direct implementation of divisive clustering, we use a top-down approach
divisive_linkage = linkage(scaled_iris, method='ward')  # Ward's method works for divisive clustering too
divisive_clusters = fcluster(divisive_linkage, t=optimal_k, criterion='maxclust')

# Add clustering results to the dataframe
iris_df['Divisive_Cluster'] = divisive_clusters

# Plot the dendrogram (for divisive method, we interpret it in reverse)
plt.figure(figsize=(12, 6))
dendrogram(divisive_linkage)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Divisive Hierarchical Clustering Dendrogram')
plt.show()

# Scatter plot visualization (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['Divisive_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Divisive Hierarchical Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#----------------------Density Based Method (DBSCAN)---------------------------------------------------

from sklearn.cluster import DBSCAN

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Default parameters, can be adjusted
iris_df['DBSCAN_Cluster'] = dbscan.fit_predict(scaled_iris)

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['DBSCAN_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('DBSCAN Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()

#-------------------------Using Model-based Method: SOM------------------------------------------------

#------------------------------------------------------------------------------------------------------



# Define the SOM grid size
som_grid_size = (3, 3)

# Initialize and train the SOM
som = MiniSom(x=som_grid_size[0], y=som_grid_size[1], input_len=scaled_iris.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(scaled_iris)
som.train_random(scaled_iris, num_iteration=500)

# Assign each data point to a cluster based on the Best Matching Unit (BMU)
bmu_indexes = np.array([som.winner(x) for x in scaled_iris])
som_clusters = [i * som_grid_size[1] + j for i, j in bmu_indexes]  

# Add clustering results to the dataframe
iris_df['SOM_Cluster'] = som_clusters

# Visualizing clusters using a scatter plot (Petal Length vs Petal Width)
plt.figure(figsize=(10, 6))
plt.scatter(iris_df['petal_length'], iris_df['petal_width'], c=iris_df['SOM_Cluster'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('SOM Clustering on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()
print ("SOM Clustering Results")
dataframe=iris_df[['species', 'SOM_Cluster']]
print (dataframe)


labels = iris_df['Cluster']

inertia = kmeans.inertia_
silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"KMeans (k={optimal_k}):")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['PAM_Cluster']

# Compute pseudo inertia
inertia = 0
for cluster in np.unique(labels):
    cluster_points = scaled_iris[labels == cluster]
    medoid = np.mean(cluster_points, axis=0)
    inertia += np.sum((cluster_points - medoid) ** 2)

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"KMedoids (PAM) (k={optimal_k}):")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['CLARA_Cluster']

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"CLARA (k={optimal_k}):")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['CLARAN_Cluster']

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"CLARAN (k={optimal_k}):")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['Agglomerative_Cluster']

# Inertia-like calculation
inertia = 0
for cluster in np.unique(labels):
    cluster_points = scaled_iris[labels == cluster]
    center = np.mean(cluster_points, axis=0)
    inertia += np.sum((cluster_points - center) ** 2)

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"Agglomerative Clustering (k={optimal_k}):")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['Divisive_Cluster']

# Inertia-like calculation
inertia = 0
for cluster in np.unique(labels):
    cluster_points = scaled_iris[labels == cluster]
    center = np.mean(cluster_points, axis=0)
    inertia += np.sum((cluster_points - center) ** 2)

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"Divisive Clustering (k={optimal_k}):")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['DBSCAN_Cluster']

# Skip inertia (not meaningful for DBSCAN)

# DBSCAN may assign -1 as noise; you need at least 2 clusters to compute silhouette
if len(set(labels)) > 1:
    silhouette = silhouette_score(scaled_iris, labels)
    calinski = calinski_harabasz_score(scaled_iris, labels)
    davies = davies_bouldin_score(scaled_iris, labels)

    print("DBSCAN:")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Index: {calinski:.3f}")
    print(f"Davies-Bouldin Index: {davies:.3f}")
else:
    print("DBSCAN found less than 2 clusters; cannot compute metrics.")
print("-" * 40)


labels = iris_df['SOM_Cluster']

# Inertia-like calculation
inertia = 0
for cluster in np.unique(labels):
    cluster_points = scaled_iris[labels == cluster]
    center = np.mean(cluster_points, axis=0)
    inertia += np.sum((cluster_points - center) ** 2)

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"SOM Clustering:")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


labels = iris_df['SOM_Cluster']

# Inertia-like calculation
inertia = 0
for cluster in np.unique(labels):
    cluster_points = scaled_iris[labels == cluster]
    center = np.mean(cluster_points, axis=0)
    inertia += np.sum((cluster_points - center) ** 2)

silhouette = silhouette_score(scaled_iris, labels)
calinski = calinski_harabasz_score(scaled_iris, labels)
davies = davies_bouldin_score(scaled_iris, labels)

print(f"SOM Clustering:")
print(f"Inertia: {inertia:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")
print(f"Davies-Bouldin Index: {davies:.3f}")
print("-" * 40)


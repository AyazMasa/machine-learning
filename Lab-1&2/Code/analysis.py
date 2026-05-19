import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import numpy as np
from IPython.display import display
from sklearn.manifold import TSNE


# Load data
df = pd.read_csv("booksData.csv", encoding='utf-8')

# 🔹 Fixing the Price column (Removing unwanted characters)
df['Price'] = df['Price'].astype(str)  # Convert to string first
df['Price'] = df['Price'].str.replace(r'[^\d.]', '', regex=True)  # Remove all non-numeric characters
df['Price'] = df['Price'].astype(float)  # Convert to float

#  Fix the Rating column (if needed)
df.rename(columns={"Rating (1-5)": "Rating"}, inplace=True)

#  Show first 5 rows (for verification)
print(df.head())

#  Summary statistics
print("\n📊 Descriptive Statistics:")
print(df.describe())

#  Handle missing values (if any)
df.dropna(inplace=True)  # Drop missing values to avoid errors

#  Save cleaned data
df.to_csv("booksData_cleaned.csv", index=False)

#  Boxplot of Book Prices
plt.figure(figsize=(8,5))
sns.boxplot(y=df['Price'], color='skyblue')
plt.title("Boxplot of Book Prices")
plt.ylabel("Price (£)")
plt.show()

# 📌 Bar Chart of Ratings
plt.figure(figsize=(8,5))
sns.countplot(x='Rating', data=df, palette="viridis")
plt.title("Distribution of Ratings")
plt.xlabel("Star Ratings")
plt.ylabel("Count")
plt.show()

#  Scatter Plot of Price vs Rating
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Rating'], y=df['Price'], color="orange", edgecolor="black")
plt.title("Scatter Plot of Price vs Rating")
plt.xlabel("Rating")
plt.ylabel("Price (£)")
plt.show()

#  Generate Word Cloud
text = " ".join(title for title in df["Title"].astype(str))

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

#  Plot the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Book Titles")
plt.show()

#  K-Means Clustering on Book Titles
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Title"].astype(str))

#  Evaluating optimal clusters using Elbow, Silhouette, and Davies-Bouldin Scores
cluster_range = range(2, 10)

wcss = []  # Within-cluster sum of squares (Elbow Method)
silhouette_scores = []  # Silhouette Score
davies_bouldin_scores = []  # Davies-Bouldin Index

for i in cluster_range:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)

    # Elbow Method (WCSS)
    wcss.append(kmeans.inertia_)

    # Silhouette Score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

    # Davies-Bouldin Index
    db_index = davies_bouldin_score(X.toarray(), kmeans.labels_)
    davies_bouldin_scores.append(db_index)

#  Plot all three evaluation methods

plt.figure(figsize=(15, 5))

#  Elbow Method (WCSS)
plt.subplot(1, 3, 1)
plt.plot(cluster_range, wcss, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

#  Silhouette Score
plt.subplot(1, 3, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--', color='g')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")

#  Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(cluster_range, davies_bouldin_scores, marker='o', linestyle='--', color='r')
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index Method")

plt.tight_layout()
plt.show()

#  Applying K-Means Clustering with Optimal Clusters
optimal_clusters = 5  # Adjust based on the elbow method graph
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

#  Extract common words from each cluster to determine genre labels
cluster_keywords = {}

for cluster in range(optimal_clusters):
    titles_in_cluster = df[df["Cluster"] == cluster]["Title"].astype(str)
    all_words = " ".join(titles_in_cluster).split()
    common_words = Counter(all_words).most_common(5)  # Top 5 common words
    cluster_keywords[cluster] = [word for word, count in common_words]

#  Assign probable genres based on common words
genre_mapping = {
    0: "Science Fiction",
    1: "Romance",
    2: "Motivation & Self-Help",
    3: "Mystery & Thriller",
    4: "History & Biographies",
}

#  Map clusters to genres
df["Genre"] = df["Cluster"].map(genre_mapping)

#  Save the book genre to a new CSV file
df.to_csv("books_with_genres.csv", index=False)
print("📂 Books with genres saved to 'books_with_genres.csv'.")

#  PCA-Based Scatter Plot for Cluster Visualization
pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X.toarray())

#  Create a scatter plot of the clusters
#  Create a scatter plot of the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    plt.scatter(
        X_reduced[df["Cluster"] == cluster, 0], 
        X_reduced[df["Cluster"] == cluster, 1], 
        label=f"Cluster {cluster} - {genre_mapping[cluster]}",
        alpha=0.6
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Book Clusters Visualization Using PCA")
plt.legend()
plt.show()

    # === Classification using MLP Classifier (Basic + Hyperparameter Tuning) ===

# Encode the genres into numerical labels
df["Genre_Label"] = df["Genre"].astype("category").cat.codes

# Convert book titles into numerical features using TF-IDF (reuse vectorizer)
X = vectorizer.fit_transform(df["Title"].astype(str))
y = df["Genre_Label"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different hyperparameter configurations for the MLP Classifier
mlp_configs = [
    {"hidden_layer_sizes": (50,), "max_iter": 200, "learning_rate_init": 0.01},
    {"hidden_layer_sizes": (100,), "max_iter": 500, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (50, 50), "max_iter": 300, "learning_rate_init": 0.005},
]

# Store results for comparison
results = []

# Train and evaluate the model for each configuration
for config in mlp_configs:
    mlp = MLPClassifier(hidden_layer_sizes=config["hidden_layer_sizes"],
                        max_iter=config["max_iter"],
                        learning_rate_init=config["learning_rate_init"],
                        random_state=42)
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        "Hidden Layers": config["hidden_layer_sizes"],
        "Max Iterations": config["max_iter"],
        "Learning Rate": config["learning_rate_init"],
        "Accuracy": accuracy
    })

# Convert results into a DataFrame for easy comparison
results_df = pd.DataFrame(results)
print("\n🔍 MLP Configuration Results:")
print(results_df)


# Print classification report for the best model (highest accuracy)
best_model_index = results_df["Accuracy"].idxmax()
best_config = mlp_configs[best_model_index]

best_mlp = MLPClassifier(hidden_layer_sizes=best_config["hidden_layer_sizes"],
                         max_iter=best_config["max_iter"],
                         learning_rate_init=best_config["learning_rate_init"],
                         random_state=42)

best_mlp.fit(X_train, y_train)
y_pred_best = best_mlp.predict(X_test)

# Print classification report
print("\n📄 Classification Report for the Best MLP Model:")
print(classification_report(y_test, y_pred_best))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=df["Genre"].astype("category").cat.categories,
            yticklabels=df["Genre"].astype("category").cat.categories)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Best MLP Model")
plt.show()

# t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=df["Genre_Label"], cmap="viridis", alpha=0.6)
plt.colorbar(scatter, label="Genre Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Book Genres")
plt.show()



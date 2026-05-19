import pandas as pd
from pathlib import Path

# Load the dataset
file_path = Path(__file__).resolve().parent / "card_transdata.csv"
df = pd.read_csv(file_path)

df_info = df.info()
df_head = df.head()

df_info, df_head

#-----------------------------------------------------------------------------------------------------------------
# Statistical summary of the dataset
df_summary = df.describe()

# Checking class distribution of the target variable (fraud)
fraud_distribution = df['fraud'].value_counts(normalize=True) * 100

# Display results
print ("Dataset Summary")
print (df_summary)
df_summary, fraud_distribution

#-----------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Plot class distribution of fraud
plt.figure(figsize=(6, 4))
df['fraud'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title("Fraud Class Distribution")
plt.xlabel("Fraud")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

#-----------------------------------------------------------------------------------------------------------------
import numpy as np

#Normaizing and Balancing minority class

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Splitting dataset into features (X) and target (y)
X = df.drop(columns=['fraud'])
y = df['fraud']

# Splitting into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert classes to NumPy array
classes = np.array([0, 1])

# Compute class weights
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Convert to dictionary format
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(class_weight_dict)

# Displaying preprocessed data summary
preprocessed_summary = pd.DataFrame(X_train_scaled).describe()
print ("Preprocessed Data Summary")
print (preprocessed_summary)

#----------------------------------------------------------------------------------------------------------------
#Train using Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize and train logistic regression model with class weights
model = LogisticRegression(class_weight=class_weight_dict, random_state=42, max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability for positive class

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Specificity calculation
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Displaying performance metrics
performance_metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"],
    "Value": [accuracy, precision, recall, specificity, f1]
})


print ("Model Performance Metrics")
print (performance_metrics)

performance_metrics

#----------------------------------------------------------------------------------------------------------------
#Visualizing the results: confusion matrix and tsne

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_test_scaled)

# Convert to DataFrame for plotting
tsne_df = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
tsne_df["Fraud"] = y_test.values

# Plot t-SNE results
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Component 1", y="Component 2", hue="Fraud", palette={0: "blue", 1: "red"}, data=tsne_df, alpha=0.5)
plt.title("t-SNE Visualization of Transactions: Logistic Regression")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Fraud", labels=["Not Fraud", "Fraud"])
plt.show()


#---------------------------------------------------------------------------------------------------------------
#Using KNN Classifier


from sklearn.neighbors import KNeighborsClassifier

# Initialize and train KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors as default
knn_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_knn = knn_model.predict(X_test_scaled)

# Calculate performance metrics
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Specificity calculation for KNN
tn_knn, fp_knn, fn_knn, tp_knn = conf_matrix_knn.ravel()
specificity_knn = tn_knn / (tn_knn + fp_knn)

# Displaying performance metrics for KNN
performance_metrics_knn = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"],
    "Value": [accuracy_knn, precision_knn, recall_knn, specificity_knn, f1_knn]
})

print ("KNN Model Performance Metrics")
print (performance_metrics_knn)

# Plot Confusion Matrix for KNN
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN Confusion Matrix")
plt.show()

# Reduce dataset size for t-SNE visualization with KNN
X_sample_knn, _, y_sample_knn, _ = train_test_split(X_test_scaled, y_test, test_size=0.9, random_state=42, stratify=y_test)

# Apply t-SNE on the smaller dataset
tsne_knn = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_sample_knn = tsne_knn.fit_transform(X_sample_knn)

# Convert to DataFrame for plotting
tsne_df_sample_knn = pd.DataFrame(X_tsne_sample_knn, columns=["Component 1", "Component 2"])
tsne_df_sample_knn["Fraud"] = y_sample_knn.values

# Plot t-SNE results for KNN
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Component 1", y="Component 2", hue="Fraud", palette={0: "blue", 1: "red"}, data=tsne_df_sample_knn, alpha=0.5)
plt.title("t-SNE Visualization of Transactions (KNN)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Fraud", labels=["Not Fraud", "Fraud"])
plt.show()


#-----------------------------------------------------------------------------------------------------------------

# DecisionTree Classifier

from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test_scaled)

# Calculate performance metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Specificity calculation
tn_dt, fp_dt, fn_dt, tp_dt = conf_matrix_dt.ravel()
specificity_dt = tn_dt / (tn_dt + fp_dt)

# Displaying performance metrics for Decision Tree
performance_metrics_dt = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"],
    "Value": [accuracy_dt, precision_dt, recall_dt, specificity_dt, f1_dt]
})

print ("Decision Tree Model Performance Metrics")
print (performance_metrics_dt)

# Plot Confusion Matrix for Decision Tree
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Reduce dataset size for t-SNE visualization with Decision Tree
X_sample_dt, _, y_sample_dt, _ = train_test_split(X_test_scaled, y_test, test_size=0.9, random_state=42, stratify=y_test)

# Apply t-SNE on the smaller dataset
tsne_dt = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_sample_dt = tsne_dt.fit_transform(X_sample_dt)

# Convert to DataFrame for plotting
tsne_df_sample_dt = pd.DataFrame(X_tsne_sample_dt, columns=["Component 1", "Component 2"])
tsne_df_sample_dt["Fraud"] = y_sample_dt.values

# Plot t-SNE results for Decision Tree
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Component 1", y="Component 2", hue="Fraud", palette={0: "blue", 1: "red"}, data=tsne_df_sample_dt, alpha=0.5)
plt.title("t-SNE Visualization of Transactions (Decision Tree)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Fraud", labels=["Not Fraud", "Fraud"])
plt.show()


#-----------------------------------------------------------------------------------------------------------------

#Support Vector Machine

from sklearn.svm import SVC


# Train SVM Classifier
svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# Calculate performance metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Specificity calculation
tn_svm, fp_svm, fn_svm, tp_svm = conf_matrix_svm.ravel()
specificity_svm = tn_svm / (tn_svm + fp_svm)

# Displaying performance metrics for SVM
performance_metrics_svm = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"],
    "Value": [accuracy_svm, precision_svm, recall_svm, specificity_svm, f1_svm]
})

print ("SVM Model Performance Metrics")
print (performance_metrics_svm)

# Plot Confusion Matrix for SVM
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
plt.show()

# Minimize dataset for t-SNE visualization with SVM
X_sample_svm, _, y_sample_svm, _ = train_test_split(X_test_scaled, y_test, test_size=0.9, random_state=42, stratify=y_test)

# Apply t-SNE on the smaller dataset
tsne_svm = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_sample_svm = tsne_svm.fit_transform(X_sample_svm)

# Convert to DataFrame for plotting
tsne_df_sample_svm = pd.DataFrame(X_tsne_sample_svm, columns=["Component 1", "Component 2"])
tsne_df_sample_svm["Fraud"] = y_sample_svm.values

# Plot t-SNE results for SVM
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Component 1", y="Component 2", hue="Fraud", palette={0: "blue", 1: "red"}, data=tsne_df_sample_svm, alpha=0.5)
plt.title("t-SNE Visualization of Transactions (SVM)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Fraud", labels=["Not Fraud", "Fraud"])
plt.show()

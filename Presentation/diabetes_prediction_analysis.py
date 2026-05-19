# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# LOAD DATA
df = pd.read_csv("diabetes.csv")

# HANDLE MISSING VALUES (Replace 0s in certain columns with NaN)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# SPLIT FEATURES AND TARGET
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DEFINE MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boost": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True),  # no random_state needed here
    "Naive Bayes": GaussianNB(),   # no random_state needed here
    "Neural Network": MLPClassifier(max_iter=1000, random_state=42)
}


# EVALUATION
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Get probability scores for ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    })

# RESULTS TABLE
results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
print("\nMODEL PERFORMANCE COMPARISON:\n")
print(results_df)

# Export results to CSV
results_df.to_csv("model_comparison.csv", index=False)


# OPTIONAL: CONFUSION MATRIX & ROC CURVE FOR BEST MODEL
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {roc_auc_score(y_test, y_proba_best):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f"ROC Curve: {best_model_name}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 1. Confusion Matrices for All Models
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.suptitle("Confusion Matrices for All Models", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrices_all_models.png")
plt.close()

# 2. Combined ROC Curves
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves for All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("roc_curves_all_models.png")
plt.close()

# 3. Bar Plot of Evaluation Metrics
metrics_df = pd.DataFrame(results).set_index("Model")
metrics_df = metrics_df.sort_values("ROC AUC", ascending=False)
metrics_df.plot(kind='bar', figsize=(12, 7))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("barplot_metrics_all_models.png")
plt.close()

# 4. Feature Importance for Gradient Boost
plt.figure(figsize=(10, 6))
importances = {}

# Extract feature importances from tree-based models
for name, model in models.items():
    if hasattr(model, "feature_importances_"):
        importances[name] = model.feature_importances_

# Create a DataFrame of feature importances
importance_df = pd.DataFrame(importances, index=X.columns)

importance_df["Gradient Boost"].sort_values().plot(kind='barh')
plt.title("Feature Importance - Gradient Boost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_gb.png")
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# 6. Boxplots of Key Features by Outcome
sns.set(style="whitegrid")

features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']

for feature in features_to_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Outcome', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Diabetes Outcome')
    plt.xlabel("Diabetes Outcome (0 = No, 1 = Yes)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(f"boxplot_{feature.lower()}.png")
    plt.close()

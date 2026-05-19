import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent / "adult.csv"
df = pd.read_csv(file_path, header=None, delimiter=",", skipinitialspace=True)

# Display basic information about the dataset
df.info(), df.head()


# Reload the dataset with the correct delimiter (comma)
df = pd.read_csv(file_path, header=None, delimiter=",", skipinitialspace=True)

# Display dataset info and first few rows
df.info(), df.head()

# Display dataset info and first few rows
df.info(), df.head()


# Assign proper column names
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Display first few rows after renaming
df.head()


# Assign proper column names based on the Adult Income dataset structure
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
df.columns = column_names

# Check for class imbalance in the target variable
class_distribution = df["income"].value_counts()

# Display class distribution
class_distribution


import matplotlib.pyplot as plt

# Plot class distribution of income
plt.figure(figsize=(6, 4))
df['income'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title("Income Class Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Plot distributions of numerical features
df.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

#--------------------------------Apply encoder to categorical variables and split dataset-------------------------
# Import the necessary module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Select categorical columns
categorical_columns = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

# Apply One-Hot Encoding
encoder = OneHotEncoder(drop="first", sparse_output=False)  # drop="first" to avoid dummy variable trap
encoded_categorical = encoder.fit_transform(df[categorical_columns])

# Convert encoded categorical values into a DataFrame
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the new one-hot encoded columns
df = df.drop(columns=categorical_columns).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# Define features (X) and target (y)
X = df.drop("income", axis=1)
y = df["income"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the first few rows of the preprocessed data
X_train.head()

#-------------------------------------Correlation Matrix---------------------------------------------------------
import seaborn as sns

# Compute correlation matrix
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

#--------------------------Check for missing values---------------------------------------------------------------
# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Display missing values
print ("Missing Values in Dataset")
print (missing_values.to_frame())

#----------------------------Checking for unique values-----------------------------------------------------------
# Check unique values in the 'income' column
unique_income_values = df['income'].unique()

# Display unique values
unique_income_values

#--------------------------Read first few lines-------------------------------------------------------------------
# Read the first few lines of the raw file to inspect its format
with open(file_path, "r") as file:
    raw_lines = [next(file) for _ in range(5)]

# Display the raw content of the file
raw_lines

#-------------------------Handling NaN----------------------------------------------------------------------------
# Attempt to reload the dataset using proper delimiter and handling quotes correctly
df = pd.read_csv(file_path, header=None, delimiter=",", quotechar='"', skipinitialspace=True, engine='python')

# Check the number of columns detected
num_columns = df.shape[1]

# Display the number of columns detected
num_columns

# Display dataset info and first few rows
df.info(), df.head()

#-----------------------Assigning names to column-----------------------------------------------------------------
# Assign proper column names
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Trim spaces from categorical values to ensure correct mapping
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert target variable to binary (0 for <=50K, 1 for >50K)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Verify if the conversion worked correctly
df['income'].unique()


#-----------------------------Identify categorical and numerical columns------------------------------------------
# Import necessary libraries for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Identify categorical and numerical columns
categorical_cols = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
numerical_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

# Apply one-hot encoding to categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Normalize numerical features to the range [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Separate features and target variable
X = df.drop(columns=["income"])
y = df["income"]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shapes of the resulting datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#--------------------Train with  Logistic Regression--------------------------------------------------------------

# Import necessary libraries for model training and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities and classes
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred_class = model.predict(X_test)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
specificity = recall_score(y_test, y_pred_class, pos_label=0)  # Specificity = Recall for class 0
f1 = f1_score(y_test, y_pred_class)
auc_score = roc_auc_score(y_test, y_pred_prob)

# Store results in a dataframe
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "Specificity", "F1-score", "ROC-AUC"],
    "Value": [accuracy, recall, precision, specificity, f1, auc_score]
})

# Display the results
print ("Logistic Regression Performance Metrics")
print (results_df)


# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

#---------------------------Apply sampling methods----------------------------------------------------------------

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to train, evaluate, and compute metrics
def train_evaluate_metrics(X_train_resampled, y_train_resampled, X_test, y_test, method_name):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Compute performance metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)  # True Negative Rate
    f1 = f1_score(y_test, y_pred)

    # Store results
    results[method_name] = [roc_auc, precision, recall, specificity, f1]

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{method_name} (AUC = {roc_auc:.2f})")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {method_name}")
    plt.show()

# Initialize dictionary to store results
results = {}

# Initialize the plot for ROC Curves
plt.figure(figsize=(8, 6))

# Baseline Model (No Resampling)
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_y_pred = baseline_model.predict(X_test)

# Compute Baseline Metrics
baseline_roc_auc = roc_auc_score(y_test, baseline_y_pred_proba)
baseline_precision = precision_score(y_test, baseline_y_pred)
baseline_recall = recall_score(y_test, baseline_y_pred)
baseline_specificity = recall_score(y_test, baseline_y_pred, pos_label=0)  # True Negative Rate
baseline_f1 = f1_score(y_test, baseline_y_pred)

# Store Baseline Results
results["Baseline"] = [baseline_roc_auc, baseline_precision, baseline_recall, baseline_specificity, baseline_f1]

# Plot Baseline ROC Curve
fpr, tpr, _ = roc_curve(y_test, baseline_y_pred_proba)
plt.plot(fpr, tpr, label=f"Baseline (AUC = {baseline_roc_auc:.2f})")

# Plot Confusion Matrix for Baseline
cm = confusion_matrix(y_test, baseline_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Baseline")
plt.show()

# Apply resampling techniques and compute metrics

# Random Oversampling
X_train_ros, y_train_ros = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_ros, y_train_ros, X_test, y_test, "Random Oversampling")

# SMOTE
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")

# Random Undersampling
X_train_rus, y_train_rus = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_rus, y_train_rus, X_test, y_test, "Random Undersampling")

# Tomek Links
X_train_tomek, y_train_tomek = TomekLinks().fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_tomek, y_train_tomek, X_test, y_test, "Tomek Links")


# Convert results to DataFrame for better readability
metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=["ROC-AUC", "Precision", "Recall", "Specificity", "F1-score"])

print("Resampling Performance Metrics:")
print(metrics_df)


#-----------Applying different sampling methods---------------------------------------------------------------


# Import necessary libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to train, evaluate, and compute metrics
def train_evaluate_metrics(X_train_resampled, y_train_resampled, X_test, y_test, method_name):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Compute performance metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)  # True Negative Rate
    f1 = f1_score(y_test, y_pred)

    # Store results
    results[method_name] = [roc_auc, precision, recall, specificity, f1]

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{method_name} (AUC = {roc_auc:.2f})")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {method_name}")
    plt.show()

# Initialize dictionary to store results
results = {}

# Initialize the plot for ROC Curves
plt.figure(figsize=(8, 6))

# Baseline Model (No Resampling)
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
baseline_y_pred = baseline_model.predict(X_test)

# Compute Baseline Metrics
baseline_roc_auc = roc_auc_score(y_test, baseline_y_pred_proba)
baseline_precision = precision_score(y_test, baseline_y_pred)
baseline_recall = recall_score(y_test, baseline_y_pred)
baseline_specificity = recall_score(y_test, baseline_y_pred, pos_label=0)  # True Negative Rate
baseline_f1 = f1_score(y_test, baseline_y_pred)

# Store Baseline Results
results["Baseline"] = [baseline_roc_auc, baseline_precision, baseline_recall, baseline_specificity, baseline_f1]

# Plot Baseline ROC Curve
fpr, tpr, _ = roc_curve(y_test, baseline_y_pred_proba)
plt.plot(fpr, tpr, label=f"Baseline (AUC = {baseline_roc_auc:.2f})")

# Plot Confusion Matrix for Baseline
cm = confusion_matrix(y_test, baseline_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Baseline")
plt.show()

# Apply resampling techniques and compute metrics

# Random Oversampling
X_train_ros, y_train_ros = RandomOverSampler(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_ros, y_train_ros, X_test, y_test, "Random Oversampling")

# SMOTE
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")

# Random Undersampling
X_train_rus, y_train_rus = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_rus, y_train_rus, X_test, y_test, "Random Undersampling")

# Tomek Links
X_train_tomek, y_train_tomek = TomekLinks().fit_resample(X_train, y_train)
train_evaluate_metrics(X_train_tomek, y_train_tomek, X_test, y_test, "Tomek Links")


# Convert results to DataFrame for better readability
metrics_df = pd.DataFrame.from_dict(results, orient='index', columns=["ROC-AUC", "Precision", "Recall", "Specificity", "F1-score"])

print("Resampling Performance Metrics:")
print(metrics_df)


#------------------Applying dimensionality reduction (LDA)--------------------------------------------------------

# Import LDA from sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Apply LDA to reduce dimensionality (LDA can have at most (num_classes - 1) components, which is 1 for binary classification)
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train logistic regression on LDA-transformed data
model_lda = LogisticRegression(max_iter=1000, random_state=42)
model_lda.fit(X_train_lda, y_train)

# Predict probabilities and classes
y_pred_prob_lda = model_lda.predict_proba(X_test_lda)[:, 1]
y_pred_class_lda = model_lda.predict(X_test_lda)

# Compute performance metrics
accuracy_lda = accuracy_score(y_test, y_pred_class_lda)
recall_lda = recall_score(y_test, y_pred_class_lda)
precision_lda = precision_score(y_test, y_pred_class_lda)
specificity_lda = recall_score(y_test, y_pred_class_lda, pos_label=0)  # Specificity = Recall for class 0
f1_lda = f1_score(y_test, y_pred_class_lda)
auc_score_lda = roc_auc_score(y_test, y_pred_prob_lda)

# Store results in a dataframe
results_lda_df = pd.DataFrame({
    "Metric": ["Accuracy", "Recall", "Precision", "Specificity", "F1-score", "ROC-AUC"],
    "Value": [accuracy_lda, recall_lda, precision_lda, specificity_lda, f1_lda, auc_score_lda]
})

# Display the results
print ("LDA Logistic Regression Performance Metrics")
print (results_lda_df)

# Compute ROC curve for LDA model
fpr_lda, tpr_lda, _ = roc_curve(y_test, y_pred_prob_lda)

# Plot ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr_lda, tpr_lda, label=f'ROC Curve (AUC = {auc_score_lda:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LDA Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Compute Confusion Matrix
cm_lda = confusion_matrix(y_test, y_pred_class_lda)

# Plot Confusion Matrix
disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda, display_labels=["<=50K", ">50K"])
disp_lda.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - LDA Logistic Regression')
plt.show()

print(df.describe())

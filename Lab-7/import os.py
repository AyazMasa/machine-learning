import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from io import BytesIO
from sklearn.preprocessing import label_binarize


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import zipfile
import os
#-----------Unzip the dataset------------------------------

# Path to the uploaded zip file
script_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(script_dir, "archive (9).zip")
extract_path = script_dir

# Extract the contents of the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List the extracted files and directories
os.listdir(extract_path)

#---------List the contents of the 'dataset'--------------------

dataset_path = os.path.join(extract_path, 'dataset')
os.listdir(dataset_path)

#-------Apply simple ML classifier to train and test---------------------

dataset_path = os.path.join(extract_path, "dataset")
img_size = (64, 64)

# Load and preprocess the images
X, y, images = [], [], []  

for breed in os.listdir(dataset_path):
    breed_path = os.path.join(dataset_path, breed)
    if os.path.isdir(breed_path):
        for img_name in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X.append(np.array(img).flatten())
                y.append(breed)
                images.append(img)
            except:
                continue

X = np.array(X)
y = np.array(y)
images = np.array(images)

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y_encoded, images, test_size=0.2, random_state=42)

# Standardize and apply PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save scaler and PCA for future use
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Define models
models = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    joblib.dump(model, f'{name}_model.pkl')
    print(f"{name} model saved as {name}_model.pkl")

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Show 5 sample predictions
    print(f"Sample predictions from {name}:")
    for i in range(5):
        plt.imshow(img_test[i])
        plt.title(f"Predicted: {le.inverse_transform([y_pred[i]])[0]} | True: {le.inverse_transform([y_test[i]])[0]}")
        plt.axis('off')
        plt.show()

#-----------------------Display metrics and plot the ROC curve--------------------------------------------------------------

img_size = (64, 64)

# Load images and labels
X, y, images = [], [], []

for breed in os.listdir(dataset_path):
    breed_path = os.path.join(dataset_path, breed)
    if os.path.isdir(breed_path):
        for img_name in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize(img_size)
                X.append(np.array(img).flatten())
                y.append(breed)
                images.append(img)
            except:
                continue

X = np.array(X)
y = np.array(y)
images = np.array(images)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_binarized = label_binarize(y_encoded, classes=np.unique(y_encoded))
n_classes = y_binarized.shape[1]

# Train-test split
X_train, X_test, y_train, y_test, yb_train, yb_test, img_train, img_test = train_test_split(
    X, y_encoded, y_binarized, images, test_size=0.2, random_state=42)


# Choose classifier
model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
model.fit(X_train_pca, yb_train)
y_prob = model.predict_proba(X_test_pca)
y_pred = model.predict(X_test_pca)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(yb_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='macro')
recall = recall_score(y_true_labels, y_pred_labels, average='macro')
f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

# Specificity for each class
specificities = []
for i in range(n_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp)
    specificities.append(specificity)
avg_specificity = np.mean(specificities)

# Print all metrics
print(f"Accuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print(f"F1-score     : {f1:.2f}")
print(f"Specificity  : {avg_specificity:.2f}")

# ROC Curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(yb_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#-----------Save the model--------------------------------

model_name = "SVM"  
model = joblib.load(f'{model_name}_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
le = joblib.load('label_encoder.pkl')


# Load image from the web
url = "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"  
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB').resize(img_size)

# Preprocess image
img_array = np.array(img).flatten().reshape(1, -1)
img_array_scaled = scaler.transform(img_array)
img_array_pca = pca.transform(img_array_scaled)

# Predict
pred_label_encoded = model.predict(img_array_pca)[0]
pred_label = le.inverse_transform([pred_label_encoded])[0]

# Show result
plt.imshow(img)
plt.title(f"Predicted Breed: {pred_label}")
plt.axis('off')
plt.show()

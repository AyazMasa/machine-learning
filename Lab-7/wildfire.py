# ---------------------- IMPORTS ----------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
import zipfile
from sklearn.model_selection import StratifiedKFold, cross_val_score
import cv2

# ---------------------- SEGMENTATION FUNCTION ----------------------
def simple_segmentation(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    segmented = cv2.bitwise_and(image_np, image_np, mask=mask)
    return segmented

# ---------------------- EXTRACT ZIP ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(script_dir, "archive (8).zip")
extract_path = script_dir
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ---------------------- LOAD DATA ----------------------
dataset_path = extract_path

img_size = (64, 64)
X, y, images = [], [], []

for split in ['train', 'valid', 'test']:
    split_path = os.path.join(dataset_path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB').resize(img_size)
                    img_np = simple_segmentation(np.array(img))
                    X.append(img_np.flatten())
                    y.append(class_name)
                    images.append(img)
                except:
                    continue

X = np.array(X)
y = np.array(y)
images = np.array(images)

# ---------------------- ENCODE & SPLIT ----------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y_encoded, images, test_size=0.2, random_state=42
)

# ---------------------- SCALING & PCA ----------------------
scaler = StandardScaler()
pca = PCA(n_components=30)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ---------------------- SAVE ARTIFACTS ----------------------
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(le, 'label_encoder.pkl')

# ---------------------- MODEL TRAINING & EVALUATION ----------------------
models = {
    "SVM": SVC(kernel='linear', C=0.5, probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    model.fit(X_train_pca, y_train)
    joblib.dump(model, f'{name}_model.pkl')
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

print(f"Sample predictions from {name}:")

# Dictionary to track how many samples we've shown per class
shown_per_class = {label: 0 for label in le.classes_}
max_per_class = 5

for i in range(len(img_test)):
    true_class = le.inverse_transform([y_test[i]])[0]
    pred_class = le.inverse_transform([y_pred[i]])[0]

    if shown_per_class[true_class] < max_per_class:
        plt.imshow(img_test[i])
        plt.title(f"Predicted: {pred_class} | True: {true_class}")
        plt.axis('off')
        plt.show()
        shown_per_class[true_class] += 1

    # Stop if all classes have reached their limit
    if all(count >= max_per_class for count in shown_per_class.values()):
        break


# ---------------------- METRICS & ROC ----------------------
yb_train = label_binarize(y_train, classes=np.unique(y_encoded))
yb_test = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = yb_train.shape[1]

model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
model.fit(X_train_pca, yb_train)

y_prob = model.predict_proba(X_test_pca)
y_pred = model.predict(X_test_pca)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(yb_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (One-vs-Rest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='macro')
recall = recall_score(y_true_labels, y_pred_labels, average='macro')
f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

specificities = []
for i in range(n_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp)
    specificities.append(specificity)
avg_specificity = np.mean(specificities)

print(f"Accuracy     : {accuracy:.2f}")
print(f"Precision    : {precision:.2f}")
print(f"Recall       : {recall:.2f}")
print(f"F1-score     : {f1:.2f}")
print(f"Specificity  : {avg_specificity:.2f}")

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(yb_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# ---------------------- WEB IMAGE PREDICTION ----------------------
model_name = "SVM"
model = joblib.load(f"{model_name}_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
le = joblib.load("label_encoder.pkl")

url = "https://upload.wikimedia.org/wikipedia/commons/0/05/Burnout_ops_on_Mangum_Fire_McCall_Smokejumpers.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize(img_size)

img_np = simple_segmentation(np.array(img))
img_array = img_np.flatten().reshape(1, -1)
img_array_scaled = scaler.transform(img_array)
img_array_pca = pca.transform(img_array_scaled)

pred_label_encoded = model.predict(img_array_pca)[0]
pred_label = le.inverse_transform([pred_label_encoded])[0]

plt.imshow(img)
plt.title(f"Predicted Class: {pred_label}")
plt.axis("off")
plt.show()

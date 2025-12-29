import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_model_affectnet_finetuned.h5"
)

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

IMG_SIZE = (197, 197)
BATCH_SIZE = 32

# -----------------------------
# LOAD MODEL
# -----------------------------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# DATA GENERATOR
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print("[INFO] Classes:", class_names)

# -----------------------------
# PREDICTIONS
# -----------------------------
print("[INFO] Running predictions...")
test_generator.reset()

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Emotion Recognition")
plt.tight_layout()
plt.show()

# -----------------------------
# F1 SCORE BAR CHART
# -----------------------------
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred
)

plt.figure(figsize=(8, 5))
plt.bar(class_names, f1)
plt.ylabel("F1-score")
plt.title("F1-score per Emotion Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

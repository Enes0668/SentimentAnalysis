import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fer2013_mini_XCEPTION.102-0.66.hdf5"
)

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "emotion_finetuned.hdf5"
)

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------
print("[INFO] Loading pretrained model...")
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# FREEZE MOST LAYERS
# -----------------------------
print("[INFO] Freezing layers...")
for layer in model.layers[:-5]:
    layer.trainable = False

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# CALLBACKS
# -----------------------------
checkpoint = ModelCheckpoint(
    OUTPUT_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
print("[INFO] Starting fine-tuning...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save(OUTPUT_MODEL_PATH)
print(f"[INFO] Model saved to: {OUTPUT_MODEL_PATH}")

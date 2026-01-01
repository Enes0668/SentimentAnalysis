import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== Ayarlar ======
PRETRAINED_MODEL = "emotion_model_affectnet.h5"
FINETUNED_MODEL = "emotion_model_affectnet_finetuned.h5"


TRAIN_DIR = "data_generated/train"
VAL_DIR = "data_generated/valid"
IMG_SIZE = (197, 197)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 7
LEARNING_RATE = 1e-4

# ====== Modeli Yükle ======
print("Önceden eğitilmiş model yükleniyor...")
base_model = load_model(PRETRAINED_MODEL, compile=False)

# ====== Katmanları Freeze Et ======
for layer in base_model.layers[:-2]:
    layer.trainable = False

# ====== Son Katmanı Yeniden Tanımla ======
x = base_model.layers[-2].output
x = Dense(128, activation='relu', name='finetune_dense')(x)
x = Dropout(0.5, name='finetune_dropout')(x)
output = Dense(NUM_CLASSES, activation='softmax', name='emotion_output')(x)  # benzersiz isim verildi
model = Model(inputs=base_model.input, outputs=output)

# ====== Modeli Derle ======
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ====== Veri Augmentation ======
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ====== Fine-Tune Eğitimi ======
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ====== Modeli Kaydet ======
model.save(FINETUNED_MODEL)
print(f"Fine-tune tamamlandı! Model kaydedildi: {FINETUNED_MODEL}")

import os
import shutil
import random

# Orijinal veri klasörleri
src_root = "data"  # data/train/images ve data/train/labels olacak
splits = ["train", "valid", "test"]

# Yeni veri klasörü
dst_root = "data_generated"
os.makedirs(dst_root, exist_ok=True)

# Kaggle AffectNet YOLO ID → emotion mapping
yolo_to_emotion = {
    0: "angry",
    1: "neutral",   # contempt sınıfını neutral sayıyoruz
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise"
}

# Veri yüzdesi (train/valid/test)
split_ratio = {
    "train": 0.7,
    "valid": 0.2,
    "test": 0.1
}

# Hedef klasörleri oluştur
for emotion in set(yolo_to_emotion.values()):
    for split in splits:
        img_dir = os.path.join(dst_root, split, emotion, "images")
        lbl_dir = os.path.join(dst_root, split, emotion, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

# Tüm resim ve label dosyalarını listele
all_files = []
img_folder = os.path.join(src_root, "train", "images")
lbl_folder = os.path.join(src_root, "train", "labels")

for img_file in os.listdir(img_folder):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(lbl_folder, label_file)
        if os.path.exists(label_path):
            all_files.append((img_file, label_file))

# Karıştır
random.shuffle(all_files)

# Split boyutlarını hesapla
total = len(all_files)
train_end = int(total * split_ratio["train"])
valid_end = train_end + int(total * split_ratio["valid"])

split_files = {
    "train": all_files[:train_end],
    "valid": all_files[train_end:valid_end],
    "test": all_files[valid_end:]
}

# Dosyaları kopyala
for split, files in split_files.items():
    for img_file, lbl_file in files:
        label_path = os.path.join(lbl_folder, lbl_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            first_class_id = int(lines[0].split()[0])
            emotion = yolo_to_emotion.get(first_class_id)
            if emotion is None:
                continue  # bilinmeyen class'ları atla

        # Hedef klasörler
        dst_img_dir = os.path.join(dst_root, split, emotion, "images")
        dst_lbl_dir = os.path.join(dst_root, split, emotion, "labels")

        # Kopyala (YOLO formatını değiştirmeden)
        shutil.copy(os.path.join(img_folder, img_file), dst_img_dir)
        shutil.copy(label_path, dst_lbl_dir)

print("Tüm veriler data_generated klasöründe train/valid/test olarak düzenlendi!")

from tensorflow.keras.models import load_model

model = load_model("ResNet-50.h5", compile=False)
model.save("emotion_model_tf12.h5")

print("Model yeniden kaydedildi")
from asl_config import ASL_CLASSES, DATA_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
import os
import json
import numpy as np
import tensorflow as tf
import cv2

# ==============================
# モデル保存設定
# ==============================
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model.{EXTENTION}")
print(f"✅ モデル保存先: {MODEL_PATH}")

# ==============================
# データローダー
# ==============================

def load_image(path, size=(64, 64)):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.resize(img, size)
    return img.astype("float32") / 255.0


def load_landmarks(json_path):
    if not os.path.exists(json_path):
        return np.zeros((21 * 3,), dtype=np.float32)
    with open(json_path, "r") as f:
        data = json.load(f)
    if not data or len(data) == 0:
        return np.zeros((21 * 3,), dtype=np.float32)
    # 1つ目の手だけを使う（複数ある場合は先頭）
    coords = np.array([[lm["x_norm"], lm["y_norm"], lm["z_norm"]] for lm in data[0]], dtype=np.float32)
    return coords.flatten()


def load_dataset(base_dir, classes, image_size=(64, 64)):
    X_img, X_skel, X_land, y = [], [], [], []
    for label_idx, cls in enumerate(classes):
        folder = os.path.join(base_dir, cls)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".jpg") and not file.endswith("_skel.jpg"):
                base_name = file[:-4]
                img_path = os.path.join(folder, f"{base_name}.jpg")
                skel_path = os.path.join(folder, f"{base_name}_skel.jpg")
                json_path = os.path.join(folder, f"{base_name}.json")

                if not os.path.exists(json_path):
                    continue

                X_img.append(load_image(img_path, image_size))
                X_skel.append(load_image(skel_path, image_size))
                X_land.append(load_landmarks(json_path))
                y.append(label_idx)

    print(f"📊 読み込み完了: {len(y)} samples")
    return np.array(X_img), np.array(X_skel), np.array(X_land), np.array(y)


# ==============================
# データセット読み込み
# ==============================
X_img, X_skel, X_land, y = load_dataset(DATA_DIR, ASL_CLASSES, IMAGE_SIZE)
num_classes = len(ASL_CLASSES)
print(f"クラス数: {num_classes} | サンプル数: {len(y)}")

# ==============================
# モデル構築
# ==============================
# エイリアスを設定（import省略の代わり）
layers = tf.keras.layers
models = tf.keras.models
Input = tf.keras.Input

# --- raw image ---
img_input = Input(shape=(*IMAGE_SIZE, 3))
x1 = layers.Conv2D(32, (3, 3), activation="relu")(img_input)
x1 = layers.MaxPooling2D()(x1)
x1 = layers.Conv2D(64, (3, 3), activation="relu")(x1)
x1 = layers.MaxPooling2D()(x1)
x1 = layers.Flatten()(x1)
x1 = layers.Dense(128, activation="relu")(x1)

# --- skeleton image ---
skel_input = Input(shape=(*IMAGE_SIZE, 3))
x2 = layers.Conv2D(32, (3, 3), activation="relu")(skel_input)
x2 = layers.MaxPooling2D()(x2)
x2 = layers.Conv2D(64, (3, 3), activation="relu")(x2)
x2 = layers.MaxPooling2D()(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(64, activation="relu")(x2)

# --- landmarks (63次元: x,y,z * 21点) ---
land_input = Input(shape=(63,))
x3 = layers.Dense(64, activation="relu")(land_input)
x3 = layers.Dense(32, activation="relu")(x3)

# --- 結合 ---
merged = layers.concatenate([x1, x2, x3])
x = layers.Dense(128, activation="relu")(merged)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=[img_input, skel_input, land_input], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ==============================
# 学習
# ==============================
history = model.fit(
    [X_img, X_skel, X_land],
    y,
    validation_split=0.2,
    epochs=15,
    batch_size=32
)

# ==============================
# 保存
# ==============================
model.save(MODEL_PATH)
print(f"✅ モデル保存完了: {MODEL_PATH}")
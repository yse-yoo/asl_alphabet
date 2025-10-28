"""
train_multimodal_asl_hands.py
ASL学習モデル (方法C: hands専用入力付き4入力構成)
"""

from asl_config import ASL_CLASSES, DATA_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
from utils.landmark_extractor import extract_landmarks_from_image
import os
import json
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

# ==============================
# 初期設定
# ==============================
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model_hands.{EXTENTION}")
print(f"✅ モデル保存先: {MODEL_PATH}")

# MediaPipe Hands初期化
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

# ==============================
# データローダー
# ==============================

def load_image(path, size=(64, 64)):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.resize(img, size)
    return img.astype("float32") / 255.0


def crop_hands_from_image(image, margin=20):
    """MediaPipe Handsで手領域を検出して切り抜き"""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    hand_crops = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1, x2 = int(max(0, min(xs) - margin)), int(min(w, max(xs) + margin))
            y1, y2 = int(max(0, min(ys) - margin)), int(min(h, max(ys) + margin))
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, IMAGE_SIZE)
                hand_crops.append(crop.astype("float32") / 255.0)

    # 手が検出されない場合はゼロ画像を返す
    if not hand_crops:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)

    # 両手がある場合 → 左右を並べる / 平均を取る
    return np.mean(np.stack(hand_crops), axis=0).astype(np.float32)


def load_landmarks(json_path):
    """Pose + Hands(最大225次元)"""
    if not os.path.exists(json_path):
        return np.zeros((225,), dtype=np.float32)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose_points, hand_points = [], []

    if "pose" in data and isinstance(data["pose"], list):
        for lm in data["pose"]:
            pose_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

    if "hands" in data and isinstance(data["hands"], list):
        for hand in data["hands"]:
            for lm in hand:
                hand_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

    combined = np.array(pose_points + hand_points, dtype=np.float32)
    if len(combined) < 225:
        combined = np.pad(combined, (0, 225 - len(combined)))
    else:
        combined = combined[:225]
    return combined


def load_dataset(base_dir, classes, image_size=(64, 64)):
    X_img, X_skel, X_land, X_hand, y = [], [], [], [], []

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

                img = cv2.imread(img_path)
                if img is None:
                    continue

                # 入力群
                X_img.append(load_image(img_path, image_size))
                X_skel.append(load_image(skel_path, image_size))
                X_land.append(load_landmarks(json_path))
                X_hand.append(crop_hands_from_image(img))
                y.append(label_idx)

    print(f"📊 読み込み完了: {len(y)} samples")
    return (
        np.array(X_img),
        np.array(X_skel),
        np.array(X_land),
        np.array(X_hand),
        np.array(y),
    )


# ==============================
# データセット読み込み
# ==============================
X_img, X_skel, X_land, X_hand, y = load_dataset(DATA_DIR, ASL_CLASSES, IMAGE_SIZE)
num_classes = len(ASL_CLASSES)
print(f"クラス数: {num_classes} | サンプル数: {len(y)}")

# ==============================
# モデル構築（4入力構成）
# ==============================
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

# --- landmarks (Pose + Hands) ---
land_input = Input(shape=(225,))
x3 = layers.Dense(128, activation="relu")(land_input)
x3 = layers.Dense(64, activation="relu")(x3)

# --- hand crop image ---
hand_input = Input(shape=(*IMAGE_SIZE, 3))
x4 = layers.Conv2D(32, (3, 3), activation="relu")(hand_input)
x4 = layers.MaxPooling2D()(x4)
x4 = layers.Conv2D(64, (3, 3), activation="relu")(x4)
x4 = layers.MaxPooling2D()(x4)
x4 = layers.Flatten()(x4)
x4 = layers.Dense(64, activation="relu")(x4)

# --- 結合 ---
merged = layers.concatenate([x1, x2, x3, x4])
x = layers.Dense(128, activation="relu")(merged)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(
    inputs=[img_input, skel_input, land_input, hand_input],
    outputs=output,
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ==============================
# 学習
# ==============================
history = model.fit(
    [X_img, X_skel, X_land, X_hand],
    y,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
)

# ==============================
# 保存
# ==============================
model.save(MODEL_PATH)
print(f"✅ モデル保存完了: {MODEL_PATH}")
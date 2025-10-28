"""
predict_words_hands.py
ASLマルチモーダルモデル（4入力: 画像 + 骨格 + ランドマーク + 手画像）推論スクリプト
"""

from asl_config import ASL_CLASSES, TEST_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
import tensorflow as tf
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import math
import mediapipe as mp

# ==============================
# 初期化
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model_hands.{EXTENTION}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ モデルが見つかりません: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES
print("✅ モデル読み込み完了")
print("クラス一覧:", class_names)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

# ==============================
# 前処理関数
# ==============================
def load_image(path, size=(64, 64)):
    """画像を読み込み＆リサイズ"""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.resize(img, size)
    return img.astype("float32") / 255.0


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


def crop_hands_from_image(image, margin=20):
    """MediaPipeで手領域を切り抜き"""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    crops = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1, x2 = int(max(0, min(xs) - margin)), int(min(w, max(xs) + margin))
            y1, y2 = int(max(0, min(ys) - margin)), int(min(h, max(ys) + margin))
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, IMAGE_SIZE)
                crops.append(crop.astype("float32") / 255.0)

    if not crops:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)

    # 両手がある場合は平均を取る
    return np.mean(np.stack(crops), axis=0).astype(np.float32)


# ==============================
# 予測関数
# ==============================
def predict_sample(base_path):
    """4入力モデルで単一画像を予測"""
    base_name = os.path.splitext(base_path)[0]
    img_path = base_name + ".jpg"
    skel_path = base_name + "_skel.jpg"
    json_path = base_name + ".json"

    img = load_image(img_path, IMAGE_SIZE)
    skel = load_image(skel_path, IMAGE_SIZE)
    land = load_landmarks(json_path)

    original = cv2.imread(img_path)
    if original is None:
        return "Error", 0.0, np.zeros_like(img)

    hand = crop_hands_from_image(original)

    preds = model.predict([
        np.expand_dims(img, axis=0),
        np.expand_dims(skel, axis=0),
        np.expand_dims(land, axis=0),
        np.expand_dims(hand, axis=0)
    ], verbose=0)

    idx = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    label = class_names[idx]
    return label, prob, img


# ==============================
# テストフォルダ全件を推論
# ==============================
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"❌ テストディレクトリが存在しません: {TEST_DIR}")

files = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg") and not f.endswith("_skel.jpg")]
cols, rows = 5, 2
per_page = cols * rows
pages = math.ceil(len(files) / per_page)

print(f"🧩 テスト画像数: {len(files)}")

for page in range(pages):
    start = page * per_page
    end = start + per_page
    batch = files[start:end]

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, fname in enumerate(batch):
        base_path = os.path.join(TEST_DIR, fname)
        pred_class, confidence, img = predict_sample(base_path)
        axes[i].imshow(img[..., ::-1])  # BGR→RGB
        axes[i].set_title(f"{fname}\n{pred_class} ({confidence:.2f})", fontsize=9)
        axes[i].axis("off")

    for j in range(len(batch), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
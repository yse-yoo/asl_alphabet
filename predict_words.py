from asl_config import ASL_CLASSES, TEST_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
import tensorflow as tf
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import math
from utils.landmark_extractor import extract_landmarks_from_image  # MediaPipe抽出を利用

# ==============================
# モデル読み込み
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model.{EXTENTION}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルファイルが存在しません: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES
print("✅ モデル読み込み完了")
print("クラス数:", len(class_names))
print("クラス一覧:", class_names)

# ==============================
# ユーティリティ関数
# ==============================
def load_image(path, size=(64, 64)):
    """画像をCNN入力形式に変換"""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*size, 3), dtype=np.float32)
    img = cv2.resize(img, size)
    return img.astype("float32") / 255.0


def flatten_landmarks(landmark_dict):
    """extract_landmarks_from_image() の出力を225次元ベクトルに変換"""
    pose_points, hand_points = [], []

    if "pose" in landmark_dict and landmark_dict["pose"]:
        for lm in landmark_dict["pose"]:
            pose_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

    if "hands" in landmark_dict and landmark_dict["hands"]:
        for hand in landmark_dict["hands"]:
            for lm in hand:
                hand_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

    combined = np.array(pose_points + hand_points, dtype=np.float32)
    if len(combined) < 225:
        combined = np.pad(combined, (0, 225 - len(combined)))
    else:
        combined = combined[:225]
    return combined


# ==============================
# 1枚を予測（JSONなし対応版）
# ==============================
def predict_sample(base_path, use_json=False):
    """
    use_json=True の場合は既存 .json を使用
    False の場合は MediaPipe でリアルタイム抽出
    """
    base_name = os.path.splitext(base_path)[0]
    img_path = base_name + ".jpg"
    skel_path = base_name + "_skel.jpg"
    json_path = base_name + ".json"

    # --- 入力画像 ---
    img = load_image(img_path, IMAGE_SIZE)
    skel = load_image(skel_path, IMAGE_SIZE)

    # --- ランドマーク ---
    if use_json and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pose_points, hand_points = [], []
        if "pose" in data:
            for lm in data["pose"]:
                pose_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])
        if "hands" in data:
            for hand in data["hands"]:
                for lm in hand:
                    hand_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])
        combined = np.array(pose_points + hand_points, dtype=np.float32)
        if len(combined) < 225:
            combined = np.pad(combined, (0, 225 - len(combined)))
        else:
            combined = combined[:225]
        X_land = combined

    else:
        # 🔹 MediaPipeで動的抽出
        original = cv2.imread(img_path)
        landmark_dict = extract_landmarks_from_image(original, mode="both")
        X_land = flatten_landmarks(landmark_dict)

    # --- 予測 ---
    preds = model.predict([
        np.expand_dims(img, axis=0),
        np.expand_dims(skel, axis=0),
        np.expand_dims(X_land, axis=0)
    ], verbose=0)

    idx = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    label = class_names[idx]
    return label, prob, img


# ==============================
# テスト画像を一括推論
# ==============================
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"テストディレクトリが存在しません: {TEST_DIR}")

files = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg") and not f.endswith("_skel.jpg")]
num_files = len(files)
cols, rows = 5, 2
per_page = cols * rows
pages = math.ceil(num_files / per_page)

print(f"🧩 テスト画像数: {num_files}")

for page in range(pages):
    start = page * per_page
    end = start + per_page
    batch = files[start:end]

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, fname in enumerate(batch):
        base_path = os.path.join(TEST_DIR, fname)
        pred_class, confidence, img = predict_sample(base_path, use_json=False)  # JSON不要で動作
        axes[i].imshow(img[..., ::-1])  # BGR→RGB
        axes[i].set_title(f"{fname}\n{pred_class} ({confidence:.2f})", fontsize=9)
        axes[i].axis("off")

    for j in range(len(batch), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

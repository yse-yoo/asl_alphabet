from asl_config import ASL_CLASSES, TEST_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE
import tensorflow as tf
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import math

# ==============================
# パラメータ
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model.{EXTENTION}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"テスト用ディレクトリが存在しません: {TEST_DIR}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルファイルが存在しません: {MODEL_PATH}")

# ==============================
# モデル読み込み
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES
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

def load_landmarks(json_path):
    """Pose + Hands (最大225次元) をロード"""
    if not os.path.exists(json_path):
        return np.zeros((225,), dtype=np.float32)

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        # 古い形式に対応
        if isinstance(data, list) and len(data) > 0:
            coords = np.array([[lm["x_norm"], lm["y_norm"], lm["z_norm"]] for lm in data[0]], dtype=np.float32)
            flat = coords.flatten()
            return np.pad(flat, (0, 225 - len(flat)))[:225]
        else:
            return np.zeros((225,), dtype=np.float32)

    pose_points, hand_points = [], []

    # Pose 33点
    if "pose" in data and isinstance(data["pose"], list):
        for lm in data["pose"]:
            pose_points.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

    # Hands 21点×N
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


# ==============================
# 1枚を予測する関数
# ==============================
def predict_sample(base_path):
    """同一ファイル名から .jpg, _skel.jpg, .json を読み込んで予測"""
    base_name = os.path.splitext(base_path)[0]
    img_path = base_name + ".jpg"
    skel_path = base_name + "_skel.jpg"
    json_path = base_name + ".json"

    # 入力読み込み
    X_img = load_image(img_path, IMAGE_SIZE)
    X_skel = load_image(skel_path, IMAGE_SIZE)
    X_land = load_landmarks(json_path)

    # 予測
    preds = model.predict([
        np.expand_dims(X_img, axis=0),
        np.expand_dims(X_skel, axis=0),
        np.expand_dims(X_land, axis=0)
    ], verbose=0)

    idx = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    label = class_names[idx]
    return label, prob, X_img


# ==============================
# テストフォルダの全サンプルを予測
# ==============================
files = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg") and not f.endswith("_skel.jpg")]
num_files = len(files)
cols = 5
rows = 2
per_page = cols * rows
pages = math.ceil(num_files / per_page)

for page in range(pages):
    start = page * per_page
    end = start + per_page
    batch = files[start:end]

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()

    for i, fname in enumerate(batch):
        base_path = os.path.join(TEST_DIR, fname)
        pred_class, confidence, img = predict_sample(base_path)

        axes[i].imshow(img[..., ::-1])  # BGR→RGB変換
        axes[i].set_title(f"{fname}\n{pred_class} ({confidence:.2f})", fontsize=10)
        axes[i].axis("off")

    for j in range(len(batch), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
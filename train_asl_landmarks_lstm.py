"""
ASL Landmark 時系列（JSON）を LSTM で学習するモデル
"""

from asl_config import ASL_CLASSES, VIDEO_DIR, MODEL_DIR, EXTENTION
import os
import json
import numpy as np
import tensorflow as tf

# ==============================
# 設定
# ==============================
CLASSES = ASL_CLASSES
DATASET_DIR = VIDEO_DIR          # あなたの保存先 (class/json)
T = 40                           # フレーム数
LAND_DIM = 225                   # pose+hands 最大225次元
BATCH_SIZE = 8
EPOCHS = 20
MODEL_PATH = os.path.join(MODEL_DIR, "asl_lstm_landmarks.h5")

# ==============================
# JSON → (T,225) に変換
# ==============================
def json_to_landmark_sequence(json_path, T=40):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])

    seq = []

    for frm in frames:
        pose_list = []
        for lm in frm.get("pose", []):
            pose_list.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

        hand_list = []
        for hand in frm.get("hands", []):
            for lm in hand:
                hand_list.extend([lm["x_norm"], lm["y_norm"], lm["z_norm"]])

        arr = np.array(pose_list + hand_list, dtype=np.float32)

        # pad or cut
        if len(arr) < LAND_DIM:
            arr = np.pad(arr, (0, LAND_DIM - len(arr)))
        else:
            arr = arr[:LAND_DIM]

        seq.append(arr)

    # Tに揃える
    if len(seq) < T:
        pad_len = T - len(seq)
        seq.extend([np.zeros(LAND_DIM)] * pad_len)
    else:
        seq = seq[:T]

    return np.array(seq, dtype=np.float32)


# ==============================
# データセット読み込み（JSON版）
# ==============================
def load_dataset(dataset_dir, classes, T=40):
    X, y = [], []

    for label_idx, cls in enumerate(classes):
        folder = os.path.join(dataset_dir, cls)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.endswith(".json"):
                continue

            json_path = os.path.join(folder, file)
            seq = json_to_landmark_sequence(json_path, T)
            X.append(seq)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Loaded dataset: {X.shape} sequences")
    return X, y


# ==============================
# モデル構築（LSTM）
# ==============================
def build_model(T=40, land_dim=225, num_classes=10):
    inputs = tf.keras.Input(shape=(T, land_dim))

    x = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(128)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ==============================
# メイン
# ==============================
if __name__ == "__main__":
    X, y = load_dataset(DATASET_DIR, CLASSES, T=T)
    num_classes = len(CLASSES)

    model = build_model(T=T, land_dim=LAND_DIM, num_classes=num_classes)

    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        shuffle=True
    )

    model.save(MODEL_PATH)
    print(f"✅ Saved model to {MODEL_PATH}")
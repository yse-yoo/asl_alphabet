"""
ASL Landmark 時系列（JSON）学習: 改訂版
- 層化分割
- zスケール調整
- Hand強調
- 早期終了/ReduceLROnPlateau
"""

from asl_config import ASL_CLASSES, VIDEO_DIR, MODEL_DIR, EXTENTION
from asl_config import T, POSE_DIM, HANDS_DIM, LAND_DIM, HAND_WEIGHT, Z_SCALE
from asl_config import BATCH_SIZE, EPOCHS, SEED
import os, json, sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ==============================
# 設定
# ==============================
CLASSES = ASL_CLASSES
DATASET_DIR = VIDEO_DIR           # データ: VIDEO_DIR/<class>/*.json

MODEL_PATH = os.path.join(MODEL_DIR, f"asl_lstm_landmarks.{EXTENTION}")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# JSON → (T,225)
# ==============================
def _frame_to_225(frm):
    # pose
    pose_list = []
    for lm in frm.get("pose", []):
        x = lm.get("x_norm", 0.0)
        y = lm.get("y_norm", 0.0)
        z = lm.get("z_norm", 0.0) * Z_SCALE
        pose_list.extend([x, y, z])

    # hands
    hand_vals = []
    for hand in frm.get("hands", []):
        for lm in hand:
            x = lm.get("x_norm", 0.0)
            y = lm.get("y_norm", 0.0)
            z = lm.get("z_norm", 0.0) * Z_SCALE
            hand_vals.extend([x, y, z])

    pose_arr = np.array(pose_list, dtype=np.float32)
    hands_arr = np.array(hand_vals, dtype=np.float32) * HAND_WEIGHT

    # 長さ調整
    if pose_arr.size < POSE_DIM:
        pose_arr = np.pad(pose_arr, (0, POSE_DIM - pose_arr.size))
    else:
        pose_arr = pose_arr[:POSE_DIM]

    if hands_arr.size < HANDS_DIM:
        hands_arr = np.pad(hands_arr, (0, HANDS_DIM - hands_arr.size))
    else:
        hands_arr = hands_arr[:HANDS_DIM]

    return np.concatenate([pose_arr, hands_arr], axis=0)  # 225


def json_to_landmark_sequence(json_path, T=40):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ JSON読み込み失敗: {json_path} ({e})")
        return np.zeros((T, LAND_DIM), dtype=np.float32)

    frames = data.get("frames", [])
    seq = []

    for frm in frames:
        arr225 = _frame_to_225(frm)
        seq.append(arr225)

    # Tに揃える（短い場合ゼロ埋め / 長い場合切り捨て）
    if len(seq) < T:
        if len(seq) < T // 3:
            # あまりに短いのは品質警告
            print(f"⚠️ 短すぎるシーケンス: {json_path} len={len(seq)} < {T//3}")
        pad_len = T - len(seq)
        seq.extend([np.zeros(LAND_DIM, dtype=np.float32)] * pad_len)
    else:
        seq = seq[:T]

    return np.array(seq, dtype=np.float32)


# ==============================
# データセット読み込み（JSON）
# ==============================
def load_dataset(dataset_dir, classes, T=40):
    X, y, paths = [], [], []

    for label_idx, cls in enumerate(classes):
        folder = os.path.join(dataset_dir, cls)
        if not os.path.isdir(folder):
            print(f"⚠️ クラスフォルダなし: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
        if len(files) == 0:
            print(f"⚠️ JSONなし: {folder}")
            continue

        files.sort()  # 安定のため
        for file in files:
            json_path = os.path.join(folder, file)
            seq = json_to_landmark_sequence(json_path, T)
            X.append(seq)
            y.append(label_idx)
            paths.append(json_path)

    X = np.array(X, dtype=np.float32)               # (N, T, 225)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Loaded: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
    return X, y, paths


# ==============================
# モデル構築（LSTM）
# ==============================
def build_model(T=40, land_dim=225, num_classes=10):
    inputs = tf.keras.Input(shape=(T, land_dim))

    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)  # ゼロ埋め対策
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ==============================
# 1バッチ過学習テスト（任意）
# ==============================
def tiny_overfit_test(model, X, y, n=16, epochs=300):
    if len(X) < n:
        print("⚠️ tiny overfit: サンプル不足でスキップ")
        return
    np.random.seed(SEED)
    idx = np.random.choice(len(X), size=n, replace=False)
    Xs, ys = X[idx], y[idx]
    print("🔬 1バッチ過学習テスト開始...")
    hist = model.fit(Xs, ys, epochs=epochs, batch_size=4, verbose=0)
    final_loss = float(hist.history["loss"][-1])
    print(f"🔬 tiny overfit 最終loss: {final_loss:.4f} （0.1前後まで落ちればOK目安）")


# ==============================
# メイン
# ==============================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y, paths = load_dataset(DATASET_DIR, CLASSES, T=T)
    num_classes = len(CLASSES)

    # ラベル検査
    u = np.unique(y)
    print("🔎 label unique:", u)
    if u.min() != 0 or u.max() != num_classes - 1:
        print("❌ ラベル範囲が不正です。ASL_CLASSES とフォルダの対応を見直してください。")
        sys.exit(1)

    # 層化分割（validation_splitの罠回避）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("✅ Split:",
          "train", X_train.shape, "val", X_val.shape)

    # モデル
    model = build_model(T=T, land_dim=LAND_DIM, num_classes=num_classes)

    # 早期終了 & 学習率減衰
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        )
    ]

    # 必要なら 1バッチ過学習テスト
    # tiny_overfit_test(model, X_train, y_train, n=16, epochs=300)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=cbs
    )

    model.save(MODEL_PATH)
    print(f"✅ Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
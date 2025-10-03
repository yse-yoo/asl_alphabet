import tensorflow as tf
import numpy as np
import os

# ==============================
# パラメータ設定
# ==============================
IMAGE_SIZE = (64, 64)   # リサイズ先
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "asl_alphabet_train"   # 学習用データセットのパス
SAVE_DIR = "models"
EXTENTION = "keras"  # "h5" or "keras"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
SAVE_PATH = os.path.join(SAVE_DIR, f"asl_model.{EXTENTION}")
print(f"✅ モデル保存先: {SAVE_PATH}")
# ==============================
# クラス名の取得（先に保存しておく）
# ==============================
class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)
print("クラス数:", num_classes)
print("クラス一覧:", class_names)

# ==============================
# データセットの読み込み
# ==============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,      # 80% train, 20% validation
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 正規化（0〜1にスケーリング）
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# パフォーマンス最適化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================
# モデル定義
# ==============================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 学習
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==============================
# 保存（SavedModel 形式）
# ==============================
model.save(SAVE_PATH)
print(f"✅ モデルを保存しました: {SAVE_PATH}")
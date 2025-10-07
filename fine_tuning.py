import os
import tensorflow as tf

# ==============================
# パラメータ設定
# ==============================
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5

BASE_MODEL_PATH = "models/asl_words_model.keras"
FINETUNE_DATA_DIR = "asl_finetune_data"
SAVE_PATH = "models/asl_model_finetuned.keras"

# ==============================
# 学習対象クラスを指定（例: A, B, nothing のみ）
# ==============================
TARGET_CLASSES = ["A", "B", "C", "nothing"]

# ==============================
# データセットの読み込み
# ==============================
if not os.path.exists(FINETUNE_DATA_DIR):
    raise FileNotFoundError(f"追加データセットが見つかりません: {FINETUNE_DATA_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    FINETUNE_DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    labels="inferred",
    label_mode="int",
    class_names=TARGET_CLASSES   # ✅ 学習させたいクラスだけ指定
)

# 正規化
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# ==============================
# モデルの読み込み
# ==============================
if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"ベースモデルが見つかりません: {BASE_MODEL_PATH}")

base_model = tf.keras.models.load_model(BASE_MODEL_PATH)

# ==============================
# Sequential モデル対応の出力層差し替え
# ==============================
if isinstance(base_model, tf.keras.Sequential):
    # 最後の層を削除
    base_model.pop()
    # 新しい出力層を追加（名前をユニークにする）
    base_model.add(tf.keras.layers.Dense(len(TARGET_CLASSES), activation="softmax", name="custom_output"))
    model = base_model
else:
    # Functional モデルなら Functional API で再構築
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    outputs = tf.keras.layers.Dense(len(TARGET_CLASSES), activation="softmax", name="custom_output")(x)
    model = tf.keras.Model(inputs, outputs)

# ==============================
# Conv層を凍結（最後の Dense 以外）
# ==============================
for layer in model.layers[:-1]:
    layer.trainable = False

# ==============================
# コンパイル
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# ファインチューニング実行
# ==============================
print("🔄 ファインチューニング開始...")
history = model.fit(train_ds, epochs=EPOCHS)

# ==============================
# 保存
# ==============================
model.save(SAVE_PATH)
print(f"✅ ファインチューニング済みモデルを保存しました: {SAVE_PATH}")

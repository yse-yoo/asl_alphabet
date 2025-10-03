import os
import tensorflow as tf

# ==============================
# パラメータ設定
# ==============================
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-5

# 学習済みモデルのパス
BASE_MODEL_PATH = "models/asl_model.keras"
# ファインチューニング用の追加データ
FINETUNE_DATA_DIR = "asl_finetune_data"
# 保存先
SAVE_PATH = "models/asl_model_finetuned.keras"

# ==============================
# データセットの読み込み
# ==============================
if not os.path.exists(FINETUNE_DATA_DIR):
    raise FileNotFoundError(f"追加データセットが見つかりません: {FINETUNE_DATA_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    FINETUNE_DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 正規化
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 高速化
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# ==============================
# 既存モデルの読み込み
# ==============================
if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"ベースモデルが見つかりません: {BASE_MODEL_PATH}")

base_model = tf.keras.models.load_model(BASE_MODEL_PATH)
base_model.summary()

# ==============================
# Conv層は凍結、Denseのみ学習
# ==============================
for layer in base_model.layers[:-2]:
    layer.trainable = False

# 再コンパイル
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# ファインチューニング実行
# ==============================
print("🔄 ファインチューニング開始...")
history = base_model.fit(
    train_ds,
    epochs=EPOCHS
)

# ==============================
# 保存
# ==============================
base_model.save(SAVE_PATH)
print(f"✅ ファインチューニング済みモデルを保存しました: {SAVE_PATH}")
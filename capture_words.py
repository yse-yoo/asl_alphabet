import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# ==============================
# パラメータ
# ==============================
IMAGE_SIZE = (64, 64)
MODEL_PATH = "models/asl_words_model.keras"
TEST_DIR = "custom_test"

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"テスト用ディレクトリが存在しません: {TEST_DIR}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルファイルが存在しません: {MODEL_PATH}")

# ==============================
# モデル読み込み
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

# 🔹 クラス名（train時と同じ順序で固定）
class_names = [
    "I_Love_You",
    "Yes",
    "No",
    "Hello",
    "Thank_You",
    "Good",
    "Sorry",
    "Please",
    "Nothing"
]

print("クラス数:", len(class_names))
print("クラス一覧:", class_names)

# ==============================
# 画像1枚を予測する関数
# ==============================
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    pred_index = np.argmax(predictions[0])
    pred_class = class_names[pred_index]
    confidence = predictions[0][pred_index]

    return pred_class, confidence, img

# ==============================
# テストディレクトリ内の画像を予測 & 表示
# ==============================
files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))]
num_files = len(files)
cols = 5
rows = math.ceil(num_files / cols)

fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
axes = axes.flatten()

for i, fname in enumerate(files):
    path = os.path.join(TEST_DIR, fname)
    pred_class, confidence, img = predict_image(path)

    axes[i].imshow(img)
    axes[i].set_title(f"{fname}\n{pred_class} ({confidence:.2f})", fontsize=10)
    axes[i].axis("off")

# 余分なサブプロットを非表示
for j in range(len(files), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
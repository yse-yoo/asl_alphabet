import os, json

DATA_DIR = "asl_alphabet_train"  # 学習時に使ったディレクトリ
SAVE_DIR = "asl_model_saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Kerasのflow_from_directoryと同じように sorted() を使う
class_names = sorted(os.listdir(DATA_DIR))

# {"A":0,"B":1,...} の形で保存
class_indices = {name: idx for idx, name in enumerate(class_names)}

CLASS_FILE = os.path.join(SAVE_DIR, "class_indices.json")
with open(CLASS_FILE, "w", encoding="utf-8") as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)

print(f"✅ JSON 書き出し完了: {CLASS_FILE}")
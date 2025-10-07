import cv2
import os
import mediapipe as mp
import tkinter as tk
from tkinter import ttk

# 保存先フォルダ
DATA_DIR = "asl_words_train"

# クラス一覧（ASL単語カテゴリ）
classes = [
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

# 各クラスの保存先フォルダを作成
for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# Tkinter GUI
# =========================
root = tk.Tk()
root.title("ASL Hand Capture")

selected_class = tk.StringVar(value=classes[0])  # 初期クラス

label = tk.Label(root, text="保存する手話単語を選択", font=("Arial", 14))
label.pack(pady=10)

combo = ttk.Combobox(root, values=classes, textvariable=selected_class, state="readonly", font=("Arial", 12))
combo.pack(pady=10)
combo.current(0)

status_label = tk.Label(root, text="未保存", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

# 保存フラグ
save_flag = tk.BooleanVar(value=False)

def trigger_save():
    save_flag.set(True)
    status_label.config(text="💾 保存要求あり", fg="green")

save_btn = tk.Button(root, text="保存 (S)", command=trigger_save, font=("Arial", 12), bg="lightgreen")
save_btn.pack(pady=10)

quit_btn = tk.Button(root, text="終了 (Q)", command=root.quit, font=("Arial", 12), bg="lightcoral")
quit_btn.pack(pady=10)

# =========================
# OpenCV カメラ処理
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()

print("✅ カメラ起動成功 (単語クラスで指定可能)")

# クラスごとに保存済み数を記録
img_counts = {}
for cls in classes:
    folder = os.path.join(DATA_DIR, cls)
    existing_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
    img_counts[cls] = len(existing_files)

def capture_loop():
    ret, frame = cap.read()
    if not ret:
        root.after(10, capture_loop)
        return

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    bbox = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            min_x, max_x = int(min(xs)) - 20, int(max(xs)) + 20
            min_y, max_y = int(min(ys)) - 20, int(max(ys)) + 20
            bbox = (min_x, min_y, max_x, max_y)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_class = selected_class.get()
    img_count = img_counts[current_class]

    cv2.putText(frame, f"Class: {current_class} | Saved: {img_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 保存処理（ボタン押されたときだけ）
    if save_flag.get() and bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w, max_x), min(h, max_y)
        crop = frame[min_y:max_y, min_x:max_x]

        save_path = os.path.join(DATA_DIR, current_class, f"{current_class}_{img_count:03d}.jpg")
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, crop)
            print(f"💾 保存: {save_path}")
            img_counts[current_class] += 1
            status_label.config(text=f"保存しました ({current_class})", fg="blue")
        else:
            print(f"⚠️ スキップ: {save_path} は既に存在します")

        save_flag.set(False)

    cv2.imshow("ASL Hand Capture", frame)
    root.after(10, capture_loop)

# 🔹キーイベント登録
def on_key(event):
    if event.keysym.lower() == 's':
        trigger_save()
    elif event.keysym.lower() == 'q':
        root.quit()

root.bind('<Key>', on_key)

# =========================
# GUI + OpenCV 並列実行
# =========================
root.after(10, capture_loop)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
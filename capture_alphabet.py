from asl_config import DATA_DIR, MARGIN
import cv2
import os
import json
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import string

# =========================
# クラス定義（A〜Z + Nothing）
# =========================
ASL_CLASSES = list(string.ascii_uppercase) + ["Nothing"]

# 保存先ディレクトリを作成
for cls in ASL_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# =========================
# MediaPipe Hands 初期化
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# Tkinter GUI 初期化
# =========================
root = tk.Tk()
root.title("ASL Hand Capture")

selected_class = tk.StringVar(value=ASL_CLASSES[0])
label = tk.Label(root, text="保存する手話クラスを選択", font=("Arial", 14))
label.pack(pady=10)

combo = ttk.Combobox(root, values=ASL_CLASSES, textvariable=selected_class, state="readonly", font=("Arial", 12))
combo.pack(pady=10)
combo.current(0)

status_label = tk.Label(root, text="未保存", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

save_flag = tk.BooleanVar(value=False)

def trigger_save():
    save_flag.set(True)
    status_label.config(text="💾 保存要求あり", fg="green")

tk.Button(root, text="保存 (S)", command=trigger_save, font=("Arial", 12), bg="lightgreen").pack(pady=10)
tk.Button(root, text="終了 (Q)", command=root.quit, font=("Arial", 12), bg="lightcoral").pack(pady=10)

# =========================
# OpenCV カメラ設定
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()

print("✅ カメラ起動成功 (raw + skeleton + JSON)")

# 各クラスの保存枚数カウント
img_counts = {cls: len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(".jpg")]) for cls in ASL_CLASSES}

# =========================
# キャプチャループ
# =========================
def capture_loop():
    ret, frame = cap.read()
    if not ret:
        root.after(10, capture_loop)
        return

    original_frame = frame.copy()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    bbox = None
    all_landmarks = []

    if results.multi_hand_landmarks:
        all_xs, all_ys = [], []
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            all_xs.extend(xs)
            all_ys.extend(ys)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = [
                {
                    "x_norm": lm.x,
                    "y_norm": lm.y,
                    "z_norm": lm.z,
                    "x_px": float(lm.x * w),
                    "y_px": float(lm.y * h)
                }
                for lm in hand_landmarks.landmark
            ]
            all_landmarks.append(coords)

        if all_xs and all_ys:
            min_x, max_x = int(min(all_xs)) - MARGIN, int(max(all_xs)) + MARGIN
            min_y, max_y = int(min(all_ys)) - MARGIN, int(max(all_ys)) + MARGIN
            bbox = (max(0, min_x), max(0, min_y), min(w, max_x), min(h, max_y))
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    current_class = selected_class.get()
    img_count = img_counts[current_class]

    # クラス名 + 保存数を表示
    cv2.putText(frame, f"Class: {current_class} | Saved: {img_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # =========================
    # 保存処理
    # =========================
    if save_flag.get():
        save_flag.set(False)
        save_base = os.path.join(DATA_DIR, current_class, f"{current_class}_{img_count:03d}")
        plain_path = f"{save_base}.jpg"
        skel_path = f"{save_base}_skel.jpg"
        json_path = f"{save_base}.json"

        # --- Nothingクラス ---
        if current_class == "Nothing":
            cv2.imwrite(plain_path, original_frame)
            cv2.imwrite(skel_path, original_frame)
            with open(json_path, "w") as f:
                json.dump([], f, indent=2)
            img_counts[current_class] += 1
            status_label.config(text=f"保存しました (Nothing)", fg="blue")
            print(f"💾 Nothing 保存: {plain_path}")
            root.after(10, capture_loop)
            return

        # --- 通常クラス ---
        if bbox is not None and all_landmarks:
            min_x, min_y, max_x, max_y = bbox
            crop_plain = original_frame[min_y:max_y, min_x:max_x]
            crop_skel = frame[min_y:max_y, min_x:max_x]

            cv2.imwrite(plain_path, crop_plain)
            cv2.imwrite(skel_path, crop_skel)
            with open(json_path, "w") as f:
                json.dump(all_landmarks, f, indent=2)

            img_counts[current_class] += 1
            status_label.config(text=f"保存しました ({current_class})", fg="blue")
            print(f"💾 保存: {plain_path}, {skel_path}, {json_path}")
        else:
            status_label.config(text=f"❌ 手が検出されません ({current_class})", fg="red")
            print(f"⚠️ {current_class}: 手が検出されませんでした")

    cv2.imshow("ASL Hand Capture", frame)
    root.after(10, capture_loop)

# =========================
# キーボード操作
# =========================
def on_key(event):
    if event.keysym.lower() == 's':
        trigger_save()
    elif event.keysym.lower() == 'q':
        root.quit()

root.bind('<Key>', on_key)
root.after(10, capture_loop)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
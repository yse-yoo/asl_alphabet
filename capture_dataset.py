import cv2
import os
import mediapipe as mp
import tkinter as tk
from tkinter import ttk

# 保存先フォルダ
DATA_DIR = "asl_finetune_data"

# クラス一覧（A〜Z + space + nothing）
classes = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["space", "nothing"]

# 各クラスの保存先フォルダを作成
for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# Tkinter GUI
# =========================
root = tk.Tk()
root.title("ASL Hand Capture")

selected_class = tk.StringVar(value="A")  # 初期クラス

label = tk.Label(root, text="保存する単語(クラス)を選択", font=("Arial", 14))
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

print("✅ カメラ起動成功 (GUIでクラス指定可能)")

img_count = 0

def capture_loop():
    global img_count

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

    # GUIで選択中のクラス
    current_class = selected_class.get()

    # 画面にクラス表示
    cv2.putText(frame, f"Class: {current_class} | Saved: {img_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # 保存処理（ボタン押されたときだけ）
    if save_flag.get() and bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w, max_x), min(h, max_y)
        crop = frame[min_y:max_y, min_x:max_x]

        save_path = os.path.join(DATA_DIR, current_class, f"{current_class}_{img_count:03d}.jpg")
        cv2.imwrite(save_path, crop)
        print(f"💾 保存: {save_path}")
        img_count += 1
        status_label.config(text=f"保存しました ({current_class})", fg="blue")
        save_flag.set(False)

    cv2.imshow("ASL Hand Capture", frame)

    # Qキーでも終了可能
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        root.quit()

    root.after(10, capture_loop)

# GUI + OpenCV 並列実行
root.after(10, capture_loop)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
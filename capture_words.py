from asl_config import ASL_CLASSES, DATA_DIR, MARGIN

import cv2
import os
import json
import mediapipe as mp
import tkinter as tk
from tkinter import ttk

# =========================
# 設定
# =========================
classes = ASL_CLASSES

# 各クラス用の保存フォルダ作成
for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# =========================
# MediaPipe 初期化
# =========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# =========================
# Tkinter GUI
# =========================
root = tk.Tk()
root.title("ASL Capture (Hands + Pose / WideBBox)")

selected_class = tk.StringVar(value=classes[0])
tk.Label(root, text="保存する手話単語を選択", font=("Arial", 14)).pack(pady=10)

combo = ttk.Combobox(root, values=classes, textvariable=selected_class, state="readonly", font=("Arial", 12))
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
# カメラ初期化
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()

print("✅ カメラ起動成功 (Hands + Pose / Wide Mode)")

# 各クラスの画像枚数カウント
img_counts = {
    cls: len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.lower().endswith(".jpg")])
    for cls in classes
}

# =========================
# キャプチャループ
# =========================
def capture_loop():
    ret, frame = cap.read()
    if not ret:
        root.after(10, capture_loop)
        return

    original = frame.copy()
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hands + Pose 同時処理
    hands_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    # Pose描画
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    all_hands, all_pose = [], []
    bbox = None
    all_xs, all_ys = [], []

    # Hands座標
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            all_xs.extend(xs)
            all_ys.extend(ys)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = [{"x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z} for lm in hand_landmarks.landmark]
            all_hands.append(coords)

    # Pose座標
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            all_xs.append(lm.x * w)
            all_ys.append(lm.y * h)
            all_pose.append({"x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z})

    # Hands + Pose 両方のbbox統合
    if all_xs and all_ys:
        min_x, max_x = int(min(all_xs)) - MARGIN, int(max(all_xs)) + MARGIN
        min_y, max_y = int(min(all_ys)) - MARGIN, int(max(all_ys)) + MARGIN
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w, max_x)
        max_y = min(h, max_y)
        bbox = (min_x, min_y, max_x, max_y)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    # 表示テキスト
    current_class = selected_class.get()
    count = img_counts[current_class]
    cv2.putText(frame, f"Class: {current_class} | Saved: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # =========================
    # 保存処理
    # =========================
    if save_flag.get():
        save_flag.set(False)
        save_base = os.path.join(DATA_DIR, current_class, f"{current_class}_{count:03d}")
        plain_path = f"{save_base}.jpg"
        skel_path = f"{save_base}_skel.jpg"
        json_path = f"{save_base}.json"

        # bboxがあれば切り抜き、なければ全体保存
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            crop_plain = original[min_y:max_y, min_x:max_x]
            crop_skel = frame[min_y:max_y, min_x:max_x]
        else:
            crop_plain = original
            crop_skel = frame

        # JSON保存データ作成
        data = {
            "pose": all_pose,
            "hands": all_hands
        }

        cv2.imwrite(plain_path, crop_plain)
        cv2.imwrite(skel_path, crop_skel)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        img_counts[current_class] += 1
        status_label.config(text=f"保存しました ({current_class})", fg="blue")
        print(f"💾 保存: {plain_path}, {skel_path}, {json_path}")

    # 次フレームへ
    cv2.imshow("ASL Capture (Hands + Pose)", frame)
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
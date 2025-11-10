from asl_config import ASL_CLASSES, VIDEO_DIR, MARGIN, HAND_CONFIDENCE, POSE_CONFIDENCE

import cv2
import os
import json
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from datetime import datetime

from utils.draw_skelton import draw_skeleton_points


# =========================
# 設定
# =========================
classes = ASL_CLASSES

# フォルダ作成
for cls in classes:
    os.makedirs(os.path.join(VIDEO_DIR, cls), exist_ok=True)

# MediaPipe 初期化
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=HAND_CONFIDENCE
)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=POSE_CONFIDENCE
)


# =========================
# Tkinter GUI
# =========================
root = tk.Tk()
root.title("ASL Video Capture (Auto Record with Countdown)")

selected_class = tk.StringVar(value=classes[0])

tk.Label(root, text="手話クラス選択", font=("Arial", 14)).pack(pady=10)
combo = ttk.Combobox(root, values=classes, textvariable=selected_class, state="readonly", font=("Arial", 12))
combo.pack(pady=10)

status_label = tk.Label(root, text="待機中", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

# カウントダウン
countdown_var = tk.StringVar(value="0")
tk.Label(root, text="カウントダウン秒数", font=("Arial", 12)).pack(pady=3)
combo_cd = ttk.Combobox(root, values=[str(i) for i in range(0, 11)],
                        textvariable=countdown_var, state="readonly", width=5, font=("Arial", 12))
combo_cd.pack(pady=3)

# 録画時間
record_seconds_var = tk.StringVar(value="3")
tk.Label(root, text="録画時間（秒）", font=("Arial", 12)).pack(pady=3)
combo_rec = ttk.Combobox(root, values=["1", "2", "3", "5", "7", "10", "15", "20"],
                         textvariable=record_seconds_var, state="readonly", width=5, font=("Arial", 12))
combo_rec.pack(pady=5)


# 状態変数
recording = False
record_frames = []
countdown = 0
countdown_start_time = None
record_start_time = None


# =========================
# 保存処理
# =========================
def save_recording():
    global record_frames
    cls = selected_class.get()

    count = len([
        f for f in os.listdir(os.path.join(VIDEO_DIR, cls))
        if f.endswith(".json")
    ])

    base = os.path.join(VIDEO_DIR, cls, f"{cls}_{count:03d}")
    json_path = base + ".json"

    with open(json_path, "w") as f:
        json.dump({"frames": record_frames}, f, indent=2)

    status_label.config(text=f"保存完了: {json_path}", fg="blue")


# =========================
# 録画開始/停止
# =========================
def toggle_record():
    global recording, record_frames, countdown, countdown_start_time, record_start_time

    if not recording:
        countdown = int(countdown_var.get())

        if countdown <= 0:
            recording = True
            record_frames = []
            record_start_time = datetime.now()
            status_label.config(text="録画中…（自動停止）", fg="red")
            return

        countdown_start_time = datetime.now()
        status_label.config(text=f"{countdown}秒後に録画開始…", fg="orange")

    else:
        recording = False
        save_recording()


# ボタン
tk.Button(root, text="録画開始 (S)", command=toggle_record, font=("Arial", 12),
          bg="lightgreen").pack(pady=10)
tk.Button(root, text="終了 (Q)", command=root.quit, font=("Arial", 12),
          bg="lightcoral").pack(pady=10)


# =========================
# カメラ初期化
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()

print("✅ カメラ起動成功")


# =========================
# メインループ
# =========================
def capture_loop():
    global recording, record_frames, countdown, countdown_start_time, record_start_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, capture_loop)
        return

    h, w, _ = frame.shape

    # ===== RGB変換 & MediaPipe =====
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose.process(rgb)
    hands_res = hands.process(rgb)

    # 描画フレーム
    draw_frame = frame.copy()
    draw_skeleton_points(draw_frame, pose_res, hands_res)

    # ===== カウントダウン =====
    if countdown > 0:
        elapsed = (datetime.now() - countdown_start_time).total_seconds()

        if elapsed >= 1:
            countdown -= 1
            countdown_start_time = datetime.now()

            if countdown > 0:
                status_label.config(text=f"{countdown}秒後に録画開始…", fg="orange")
            else:
                recording = True
                record_frames = []
                record_start_time = datetime.now()
                status_label.config(text="録画中…（自動停止）", fg="red")

        cv2.putText(draw_frame, f"{countdown}",
                    (w // 2 - 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,
                    (0, 0, 255), 6)

    # ===== 録画処理 =====
    if recording and countdown == 0:
        duration = int(record_seconds_var.get())
        elapsed_rec = (datetime.now() - record_start_time).total_seconds()

        if elapsed_rec >= duration:
            recording = False
            save_recording()
        else:

            frame_data = {"pose": [], "hands": []}

            if pose_res.pose_landmarks:
                frame_data["pose"] = [
                    {"x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z}
                    for lm in pose_res.pose_landmarks.landmark
                ]

            if hands_res.multi_hand_landmarks:
                frame_data["hands"] = [
                    [
                        {"x_norm": lm.x, "y_norm": lm.y, "z_norm": lm.z}
                        for lm in hl.landmark
                    ]
                    for hl in hands_res.multi_hand_landmarks
                ]

            record_frames.append(frame_data)

    # ===== 表示 =====
    cv2.imshow("ASL Video Capture", draw_frame)

    # 次フレームへ
    root.after(10, capture_loop)


# キー操作
def on_key(event):
    if event.keysym.lower() == 's':
        toggle_record()
    elif event.keysym.lower() == 'q':
        root.quit()

root.bind("<Key>", on_key)


# 実行
root.after(10, capture_loop)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
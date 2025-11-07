from asl_config import ASL_CLASSES, VIDEO_DIR, MARGIN

import cv2
import os
import json
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# =========================
# 設定
# =========================
classes = ASL_CLASSES

# 各クラス保存フォルダ作成
for cls in classes:
    os.makedirs(os.path.join(VIDEO_DIR, cls), exist_ok=True)

# MediaPipe初期化
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
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
root.title("ASL Video Capture (Landmark Time-Series)")

selected_class = tk.StringVar(value=classes[0])
tk.Label(root, text="手話クラス選択", font=("Arial", 14)).pack(pady=10)

combo = ttk.Combobox(root, values=classes, textvariable=selected_class, state="readonly", font=("Arial", 12))
combo.pack(pady=10)
combo.current(0)

status_label = tk.Label(root, text="待機中", font=("Arial", 12), fg="blue")
status_label.pack(pady=10)

# 録画状態
recording = False
record_frames = []  # 録画中のランドマーク時系列
video_writer = None


def toggle_record():
    global recording, record_frames, video_writer

    if not recording:
        # 開始
        recording = True
        record_frames = []
        status_label.config(text="録画中… (Sで停止)", fg="red")
    else:
        # 停止 → 保存処理
        recording = False
        status_label.config(text="保存中…", fg="green")

        cls = selected_class.get()

        # JSONの数をカウントして次の番号を決める
        count = len([
            f for f in os.listdir(os.path.join(VIDEO_DIR, cls))
            if f.endswith(".json")
        ])

        base = os.path.join(VIDEO_DIR, cls, f"{cls}_{count:03d}")
        json_path = base + ".json"

        # JSON保存
        with open(json_path, "w") as f:
            json.dump({"frames": record_frames}, f, indent=2)

        status_label.config(text=f"保存完了: {json_path}", fg="blue")


tk.Button(root, text="録画開始/停止 (S)", command=toggle_record, font=("Arial", 12), bg="lightgreen").pack(pady=10)
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

print("✅ カメラ起動成功")


# =========================
# メインループ
# =========================
def capture_loop():
    global recording, record_frames

    ret, frame = cap.read()
    if not ret:
        root.after(10, capture_loop)
        return

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe処理
    hands_res = hands.process(rgb)
    pose_res = pose.process(rgb)

    # 描画用コピー
    draw_frame = frame.copy()

    # 描画
    if pose_res.pose_landmarks:
        mp_draw.draw_landmarks(draw_frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hands_res.multi_hand_landmarks:
        for hl in hands_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(draw_frame, hl, mp_hands.HAND_CONNECTIONS)

    # =========================
    # 録画処理
    # =========================
    if recording:
        frame_data = {
            "pose": [],
            "hands": []
        }

        # Pose landmarks
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                frame_data["pose"].append({
                    "x_norm": lm.x,
                    "y_norm": lm.y,
                    "z_norm": lm.z,
                })

        # Hands landmarks
        if hands_res.multi_hand_landmarks:
            hands_list = []
            for hl in hands_res.multi_hand_landmarks:
                coords = []
                for lm in hl.landmark:
                    coords.append({
                        "x_norm": lm.x,
                        "y_norm": lm.y,
                        "z_norm": lm.z
                    })
                hands_list.append(coords)
            frame_data["hands"] = hands_list

        # フレーム記録
        record_frames.append(frame_data)

    # 表示
    cv2.imshow("ASL Video Capture", draw_frame)
    root.after(10, capture_loop)


# =========================
# キー
# =========================
def on_key(event):
    if event.keysym.lower() == 's':
        toggle_record()
    elif event.keysym.lower() == 'q':
        root.quit()

root.bind("<Key>", on_key)
root.after(10, capture_loop)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
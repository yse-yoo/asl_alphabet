"""
動き検出 + スムージング + スライドウィンドウ
学習済み LSTM (T x 225) モデルを用いた、WebカメラからのリアルタイムASL予測
"""

from asl_config import ASL_CLASSES, MODEL_DIR
from asl_config import PROB_THRESH, PRED_SMOOTH, T, LAND_DIM
from asl_config import START_MOV_THRESH, STOP_MOV_THRESH, START_FRAMES, STOP_FRAMES
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

# =========================
# 設定（必要に応じて調整）
# =========================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_lstm_landmarks.h5")
CLASSES = ASL_CLASSES

# 動き検出（C方式のキモ：ヒステリシスで開始・終了を安定化）
# movement_score = ||vec_t - vec_(t-1)|| / sqrt(LAND_DIM)

# 表示
DRAW_LANDMARKS = True       # 画面にスケルトンを描画
SHOW_FPS = True             # FPS表示

# =========================
# モデル読み込み
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# =========================
# Mediapipe 初期化
# =========================
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.65
)

# =========================
# 1フレーム → (225,) ベクトル
# =========================
def extract_landmark_vec(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pose_res  = pose.process(rgb)
    hands_res = hands.process(rgb)

    pose_list = []
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            pose_list.extend([lm.x, lm.y, lm.z])

    hand_list = []
    if hands_res.multi_hand_landmarks:
        for hand in hands_res.multi_hand_landmarks:
            for lm in hand.landmark:
                hand_list.extend([lm.x, lm.y, lm.z])

    arr = np.array(pose_list + hand_list, dtype=np.float32)
    if arr.size < LAND_DIM:
        arr = np.pad(arr, (0, LAND_DIM - arr.size))
    else:
        arr = arr[:LAND_DIM]
    return arr

# =========================
# ユーティリティ（描画）
# =========================
def draw_overlays(frame, label, prob, state, mov, fps=None):
    h, w = frame.shape[:2]

    # 状態表示
    state_text = f"STATE: {state}  MOV:{mov:.4f}"
    cv2.putText(frame, state_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 255), 2)

    # ラベル表示
    if label is not None:
        txt = f"{label} ({prob:.2f})" if prob is not None else label
        cv2.putText(frame, txt, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 240, 60), 2)
    else:
        cv2.putText(frame, "...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160, 160, 160), 2)

    if SHOW_FPS and fps is not None:
        cv2.putText(frame, f"FPS:{fps:.1f}", (w - 150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 60), 2)

# =========================
# メインループ
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラが開けません")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # スライドウィンドウ（最新Tフレーム）
    landmark_buffer: deque[np.ndarray] = deque(maxlen=T)

    # 予測のスムージング
    preds_buffer: deque[np.ndarray] = deque(maxlen=PRED_SMOOTH)

    # 動き検出ヒステリシス
    # True: 動作中（推論ON） / False: 静止（推論OFF）
    active = False
    start_cnt = 0
    stop_cnt  = 0
    prev_vec  = None

    # 速度測定
    t_prev = time.time()
    fps = None

    print("✅ Webカメラ起動（qで終了）")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ランドマーク抽出
        vec = extract_landmark_vec(frame)
        landmark_buffer.append(vec)

        # 動き量（正規化L2）
        if prev_vec is None:
            movement = 0.0
        else:
            movement = np.linalg.norm(vec - prev_vec) / (LAND_DIM ** 0.5)
        prev_vec = vec

        # ヒステリシスで状態遷移
        if active:
            if movement < STOP_MOV_THRESH:
                stop_cnt += 1
            else:
                stop_cnt = 0
            if stop_cnt >= STOP_FRAMES:
                active = False
                preds_buffer.clear()  # 推論履歴クリア
        else:
            if movement > START_MOV_THRESH:
                start_cnt += 1
            else:
                start_cnt = 0
            if start_cnt >= START_FRAMES:
                active = True
                preds_buffer.clear()  # 新しい動作へ

        # 描画用スケルトン
        if DRAW_LANDMARKS:
            # 再処理は重いので簡易描画だけ
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res  = pose.process(rgb)
            hands_res = hands.process(rgb)
            if pose_res.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if hands_res.multi_hand_landmarks:
                for hl in hands_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # 予測（条件: バッファ満杯 & 動作中）
        label = None
        prob  = None
        if active and len(landmark_buffer) == T:
            inp = np.array(landmark_buffer, dtype=np.float32).reshape(1, T, LAND_DIM)
            pred = model.predict(inp, verbose=0)[0]  # shape: (num_classes,)
            preds_buffer.append(pred)

            # スムージング
            avg_pred = np.mean(np.stack(preds_buffer, axis=0), axis=0)
            idx = int(np.argmax(avg_pred))
            prob = float(avg_pred[idx])

            if prob >= PROB_THRESH:
                label = CLASSES[idx]
            else:
                label = "..."

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = 1.0 / dt

        draw_overlays(
            frame,
            label=label,
            prob=prob,
            state="ACTIVE" if active else "IDLE",
            mov=movement,
            fps=fps
        )

        cv2.imshow("ASL Realtime (C-Mode)", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 終了しました")


if __name__ == "__main__":
    main()
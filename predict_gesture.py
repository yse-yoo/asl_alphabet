"""
ASL LSTM 時系列モデル（225次元）リアルタイム予測
トレーニングコードと完全一致した Landmark 抽出処理
"""

from asl_config import (
    ASL_CLASSES, MODEL_DIR, GESTURE_MODEL,
    PROB_THRESH, PRED_SMOOTH, T, LAND_DIM,
    START_MOV_THRESH, STOP_MOV_THRESH, START_FRAMES, STOP_FRAMES,
    HAND_WEIGHT, Z_SCALE, EXTENTION, POSE_DIM, HANDS_DIM,
    HAND_CONFIDENCE, POSE_CONFIDENCE,
    VIDEO_WIDTH, VIDEO_HEIGHT
)

import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time

# =====================================================
# モデル読み込み
# =====================================================
MODEL_PATH = os.path.join(MODEL_DIR, GESTURE_MODEL)
model = tf.keras.models.load_model(MODEL_PATH)
CLASSES = ASL_CLASSES

print(f"✅ Loaded model: {MODEL_PATH}")

# =====================================================
# Mediapipe
# =====================================================
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=HAND_CONFIDENCE
)
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=POSE_CONFIDENCE
)

# =====================================================
# 1フレーム → (225,) ベクトル（train と完全一致）
# =====================================================
def extract_landmark_vec(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pose_res  = pose.process(rgb)
    hands_res = hands.process(rgb)

    # ----- Pose -----
    pose_list = []
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks.landmark:
            pose_list.extend([
                lm.x,
                lm.y,
                lm.z * Z_SCALE
            ])
    pose_arr = np.array(pose_list, dtype=np.float32)
    if pose_arr.size < POSE_DIM:
        pose_arr = np.pad(pose_arr, (0, POSE_DIM - pose_arr.size))
    else:
        pose_arr = pose_arr[:POSE_DIM]

    # ----- Hands -----
    hand_vals = []
    if hands_res.multi_hand_landmarks:
        for hand in hands_res.multi_hand_landmarks:
            for lm in hand.landmark:
                hand_vals.extend([
                    lm.x,
                    lm.y,
                    lm.z * Z_SCALE
                ])
    hands_arr = np.array(hand_vals, dtype=np.float32) * HAND_WEIGHT
    if hands_arr.size < HANDS_DIM:
        hands_arr = np.pad(hands_arr, (0, HANDS_DIM - hands_arr.size))
    else:
        hands_arr = hands_arr[:HANDS_DIM]

    # ----- combine -----
    vec225 = np.concatenate([pose_arr, hands_arr], axis=0)
    return vec225.astype(np.float32)


# =====================================================
# 描画ユーティリティ
# =====================================================
def draw_texts(frame, label, prob, state, mov, fps):
    h, w = frame.shape[:2]

    # cv2.putText(frame, f"STATE:{state}  MOV:{mov:.4f}", (10, 28),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,255), 2)

    if label:
        cv2.putText(frame, f"{label} ({prob:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,255,80), 2)
    else:
        cv2.putText(frame, "...", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (140,140,140), 2)

    # cv2.putText(frame, f"FPS:{fps:.1f}", (w - 160, 28),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250,220,80), 2)


# =====================================================
# メインループ（Cモード）
# =====================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラが開けない")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    landmark_buf = deque(maxlen=T)
    preds_buf    = deque(maxlen=PRED_SMOOTH)

    active = False
    start_cnt = 0
    stop_cnt  = 0
    prev_vec  = None

    t_prev = time.time()
    print("✅ Start (qで終了)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 225次元ベクトル抽出 ----
        vec = extract_landmark_vec(frame)
        landmark_buf.append(vec)

        # ---- movement ----
        if prev_vec is None:
            movement = 0.0
        else:
            movement = np.linalg.norm(vec - prev_vec) / (LAND_DIM**0.5)
        prev_vec = vec

        # ---- ヒステリシス ----
        if active:
            if movement < STOP_MOV_THRESH:
                stop_cnt += 1
            else:
                stop_cnt = 0

            if stop_cnt >= STOP_FRAMES:
                active = False
                preds_buf.clear()

        else:
            if movement > START_MOV_THRESH:
                start_cnt += 1
            else:
                start_cnt = 0

            if start_cnt >= START_FRAMES:
                active = True
                preds_buf.clear()

        # ---- 予測 ----
        label = None
        prob  = None

        if active and len(landmark_buf) == T:
            seq = np.array(landmark_buf, dtype=np.float32).reshape(1, T, LAND_DIM)
            pred = model.predict(seq, verbose=0)[0]
            preds_buf.append(pred)

            avg_pred = np.mean(np.stack(preds_buf), axis=0)
            idx = int(np.argmax(avg_pred))
            prob = float(avg_pred[idx])

            if prob >= PROB_THRESH:
                label = CLASSES[idx]
            else:
                label = None

        # ---- スケルトン描画 ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        p_res = pose.process(rgb)
        h_res = hands.process(rgb)
        if p_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, p_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if h_res.multi_hand_landmarks:
            for hl in h_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # ---- FPS ----
        t_now = time.time()
        fps = 1.0 / (t_now - t_prev)
        t_prev = t_now

        draw_texts(frame, label, prob, "ACTIVE" if active else "IDLE", movement, fps)

        # ---- show ----
        cv2.imshow("ASL LSTM Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
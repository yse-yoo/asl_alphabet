from asl_config import ASL_CLASSES, MODEL_DIR, EXTENTION, IMAGE_SIZE
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# =========================
# モデル読み込み
# =========================
MODEL_PATH = f"{MODEL_DIR}/asl_multimodal_model.{EXTENTION}"
print(f"✅ モデル読込中: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ モデル読み込み完了")

# =========================
# MediaPipe 初期化
# =========================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# =========================
# ユーティリティ
# =========================
def preprocess_image(frame, size=(64, 64)):
    """画像をCNN入力サイズに変換"""
    img = cv2.resize(frame, size)
    return img.astype("float32") / 255.0

def extract_landmarks(results_hands, results_pose, w, h):
    """Pose(33点) + Hands(最大2×21点) → 225次元ベクトル"""
    all_pose, all_hands = [], []

    # Pose
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            all_pose.extend([lm.x, lm.y, lm.z])

    # Hands
    if results_hands.multi_hand_landmarks:
        for hand in results_hands.multi_hand_landmarks:
            for lm in hand.landmark:
                all_hands.extend([lm.x, lm.y, lm.z])

    combined = np.array(all_pose + all_hands, dtype=np.float32)
    if len(combined) < 225:
        combined = np.pad(combined, (0, 225 - len(combined)))
    else:
        combined = combined[:225]

    return combined

# =========================
# カメラ起動
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()
print("✅ カメラ起動成功")

# =========================
# ループ
# =========================
print("🟢 推論開始 (Qで終了)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe推論
    results_hands = hands.process(rgb)
    results_pose = pose.process(rgb)

    # 骨格描画
    if results_pose.pose_landmarks:
        mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # 入力整形
    img_input = preprocess_image(frame, IMAGE_SIZE)
    skel_input = img_input.copy()  # 今は同じ映像を使う（必要なら別描画に変更可）
    land_input = extract_landmarks(results_hands, results_pose, w, h)

    # 推論
    pred = model.predict([
        np.expand_dims(img_input, axis=0),
        np.expand_dims(skel_input, axis=0),
        np.expand_dims(land_input, axis=0)
    ], verbose=0)

    idx = int(np.argmax(pred))
    prob = float(np.max(pred))
    label = ASL_CLASSES[idx]

    # 画面表示
    cv2.putText(frame, f"{label} ({prob*100:.1f}%)",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 0) if prob > 0.7 else (0, 165, 255), 3)

    cv2.imshow("ASL Prediction (Hands + Pose)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
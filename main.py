from asl_config import ASL_CLASSES, DATA_DIR, MODEL_DIR, EXTENTION, IMAGE_SIZE, MARGIN
import os, io, json
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import mediapipe as mp

# ==============================
# モデル & クラス名の読み込み
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_multimodal_model.{EXTENTION}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES

# ==============================
# MediaPipe 初期化
# ==============================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# ==============================
# FastAPI
# ==============================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================
# データ前処理関数
# ==============================
def preprocess_image(img: np.ndarray):
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype("float32") / 255.0

def extract_landmarks_from_image(image: np.ndarray):
    """Pose(33) + Hands(最大42) → 225次元に整形"""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)

    all_pose, all_hands = [], []

    # Pose
    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            all_pose.extend([lm.x, lm.y, lm.z])

    # Hands
    if hands_results.multi_hand_landmarks:
        for hand in hands_results.multi_hand_landmarks:
            for lm in hand.landmark:
                all_hands.extend([lm.x, lm.y, lm.z])

    combined = np.array(all_pose + all_hands, dtype=np.float32)
    if len(combined) < 225:
        combined = np.pad(combined, (0, 225 - len(combined)))
    else:
        combined = combined[:225]
    return combined

# ==============================
# 推論関数
# ==============================
def predict_multimodal(img: np.ndarray):
    # 通常画像とスケルトン画像を作成
    skel = img.copy()

    # MediaPipe描画
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)
    mp.solutions.drawing_utils.draw_landmarks(
        skel, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    ) if pose_results.pose_landmarks else None
    if hands_results.multi_hand_landmarks:
        for hand in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                skel, hand, mp_hands.HAND_CONNECTIONS
            )

    # 前処理
    X_img = preprocess_image(img)
    X_skel = preprocess_image(skel)
    X_land = extract_landmarks_from_image(img)

    # 予測
    preds = model.predict([
        np.expand_dims(X_img, axis=0),
        np.expand_dims(X_skel, axis=0),
        np.expand_dims(X_land, axis=0)
    ], verbose=0)

    idx = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    return class_names[idx], prob

# ==============================
# ルート
# ==============================
@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/app")
async def camera_page():
    with open("static/app.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ==============================
# 予測エンドポイント
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # OpenCVで読み込み
    img = cv2.imread(save_path)
    if img is None:
        return {"error": "画像が読み込めません"}

    pred_class, confidence = predict_multimodal(img)

    return {
        "filename": file.filename,
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 2),
        "image_url": f"/static/uploads/{file.filename}"
    }

@app.get("/test")
async def index():
    with open("static/test.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
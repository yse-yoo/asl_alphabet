from asl_config import ASL_CLASSES, USE_MODEL, MODEL_DIR, EXTENTION, IMAGE_SIZE, MARGIN
from utils.labels import normalize_label
import os, io, json
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import mediapipe as mp

# ==============================
# モデル & クラス名の読み込み
# ==============================
MODEL_PATH = os.path.join(MODEL_DIR, f"{USE_MODEL}.{EXTENTION}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"モデルが見つかりません: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ASL_CLASSES
input_count = len(model.inputs)
print(f"✅ モデル読み込み完了: {MODEL_PATH}")
print(f"🔹 入力数: {input_count}")

# ==============================
# MediaPipe 初期化
# ==============================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)

    all_pose, all_hands = [], []

    if pose_results.pose_landmarks:
        for lm in pose_results.pose_landmarks.landmark:
            all_pose.extend([lm.x, lm.y, lm.z])

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


def make_skeleton_image(img: np.ndarray):
    """ポーズと手を重ねてスケルトン画像作成"""
    skel = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(skel, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if hands_results.multi_hand_landmarks:
        for hand in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(skel, hand, mp_hands.HAND_CONNECTIONS)
    return skel


def crop_hands_from_image(image: np.ndarray, margin=20):
    """手領域を切り抜き平均化（4入力モデル用）"""
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    crops = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1, x2 = int(max(0, min(xs) - margin)), int(min(w, max(xs) + margin))
            y1, y2 = int(max(0, min(ys) - margin)), int(min(h, max(ys) + margin))
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, IMAGE_SIZE)
                crops.append(crop.astype("float32") / 255.0)
    if not crops:
        return np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
    return np.mean(np.stack(crops), axis=0).astype(np.float32)

# ==============================
# 推論関数
# ==============================
def predict_from_modal(X_img, X_skel, X_land, X_hand=None):
    if input_count == 4:
        preds = model.predict([
            np.expand_dims(X_img, axis=0),
            np.expand_dims(X_skel, axis=0),
            np.expand_dims(X_land, axis=0),
            np.expand_dims(X_hand, axis=0)
        ], verbose=0)
    else:
        preds = model.predict([
            np.expand_dims(X_img, axis=0),
            np.expand_dims(X_skel, axis=0),
            np.expand_dims(X_land, axis=0)
        ], verbose=0)

    idx = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    return class_names[idx], prob


def predict_multimodal(img: np.ndarray):
    """画像から自動的に手やポーズを抽出して予測"""
    skel = make_skeleton_image(img)
    X_img = preprocess_image(img)
    X_skel = preprocess_image(skel)
    X_land = extract_landmarks_from_image(img)

    if input_count == 4:
        X_hand = crop_hands_from_image(img)
    else:
        X_hand = None

    pred_class, confidence = predict_from_modal(X_img, X_skel, X_land, X_hand)
    return pred_class, confidence

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

@app.get("/camera")
async def camera_page2():
    with open("static/camera.html", "r", encoding="utf-8") as f:
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

    img = cv2.imread(save_path)
    if img is None:
        return {"error": "画像が読み込めません"}

    pred_class, confidence = predict_multimodal(img)

    return {
        "filename": file.filename,
        "predicted_class": pred_class,
        "label": normalize_label(pred_class),
        "confidence": round(float(confidence), 2),
        "image_url": f"/static/uploads/{file.filename}"
    }

# ==============================
# 画像アップロード（/predict/image）
# ==============================
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    img = cv2.imread(save_path)
    if img is None:
        return JSONResponse({"error": "画像が読み込めません"}, status_code=400)

    skel = make_skeleton_image(img)
    X_img = preprocess_image(img)
    X_skel = preprocess_image(skel)
    X_land = extract_landmarks_from_image(img)
    X_hand = crop_hands_from_image(img) if input_count == 4 else None

    pred_class, confidence = predict_from_modal(X_img, X_skel, X_land, X_hand)
    return {
        "type": "image",
        "filename": file.filename,
        "predicted_class": pred_class,
        "label": normalize_label(pred_class),
        "confidence": round(confidence, 3)
    }

# ==============================
# JSON入力（Pose + Hands座標）
# ==============================
@app.post("/predict/json")
async def predict_json(data: dict = Body(...)):
    if not isinstance(data, dict):
        return JSONResponse({"error": "JSON形式が不正です"}, status_code=400)

    pose_points, hand_points = [], []
    if "pose" in data and isinstance(data["pose"], list):
        for lm in data["pose"]:
            pose_points.extend([lm["x_norm"], lm["y_norm"], lm.get("z_norm", 0.0)])
    if "hands" in data and isinstance(data["hands"], list):
        for hand in data["hands"]:
            for lm in hand:
                hand_points.extend([lm["x_norm"], lm["y_norm"], lm.get("z_norm", 0.0)])

    combined = np.array(pose_points + hand_points, dtype=np.float32)
    if len(combined) < 225:
        combined = np.pad(combined, (0, 225 - len(combined)))
    else:
        combined = combined[:225]

    dummy = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32)
    X_hand = np.zeros((*IMAGE_SIZE, 3), dtype=np.float32) if input_count == 4 else None

    pred_class, confidence = predict_from_modal(dummy, dummy, combined, X_hand)
    return {
        "type": "json",
        "predicted_class": pred_class,
        "label": normalize_label(pred_class),
        "confidence": round(confidence, 3)
    }

@app.get("/test")
async def test_page():
    with open("static/test.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
import uvicorn
import os
import json
import numpy as np
import tensorflow as tf
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

from asl_config import (
    ASL_CLASSES, T, LAND_DIM,
    MODEL_DIR, EXTENTION
)

# ===============================
# ✅ モデル読み込み
# ===============================
MODEL_PATH = os.path.join(MODEL_DIR, f"asl_lstm_landmarks.{EXTENTION}")
print("✅ Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ✅ 推論（CPUブロックを async 化）
# ===============================
loop = asyncio.get_event_loop()

def run_predict(buffer):
    x = np.array(buffer, dtype=np.float32).reshape(1, T, LAND_DIM)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    return ASL_CLASSES[idx], float(pred[idx])

async def predict_async(buffer):
    return await loop.run_in_executor(None, lambda: run_predict(buffer))

# ===============================
# ✅ WebSocket
# ===============================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected")

    buffer = deque(maxlen=T)

    while True:
        try:
            msg = await ws.receive_text()
        except Exception:
            print("❌ Client disconnected (receive error)")
            break

        # ----------------------
        # JSON パース
        # ----------------------
        try:
            obj = json.loads(msg)
            vec = obj.get("landmark", [])
        except:
            continue

        # ----------------------
        # バッファ更新
        # ----------------------
        if len(vec) == LAND_DIM:
            buffer.append(vec)

        result = {
            "ready": False,
            "label": "...",
            "prob": 0.0
        }

        # ----------------------
        # 推論実行
        # ----------------------
        if len(buffer) == T:
            try:
                label, prob = await predict_async(buffer)
                result.update({
                    "ready": True,
                    "label": label,
                    "prob": prob
                })
            except Exception as e:
                print("Predict error:", e)
                continue

        # ----------------------
        # 安全な送信
        # ----------------------
        try:
            await ws.send_text(json.dumps(result))
        except Exception:
            print("❌ Client disconnected during send")
            break

    print("🔚 WebSocket closed")

# ===============================
# ✅ 実行
# ===============================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

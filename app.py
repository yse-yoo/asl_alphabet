import uvicorn
import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from collections import deque
from asl_config import (
    ASL_CLASSES, T, LAND_DIM, MODEL_DIR, EXTENTION,
    HAND_WEIGHT, Z_SCALE
)

MODEL_PATH = os.path.join(MODEL_DIR, f"asl_lstm_landmarks.{EXTENTION}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

app = FastAPI()

# CORSの許可設定（WebSocketにも効く）
# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://localhost:5500",
#     "http://127.0.0.1:5500",
#     "http://localhost:8000",
#     "http://127.0.0.1:8000",
# ]
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# クライアントごとにバッファを保持
client_buffers = {}

def predict_sequence(buffer):
    x = np.array(buffer, dtype=np.float32).reshape(1, T, LAND_DIM)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    prob = float(pred[idx])
    return ASL_CLASSES[idx], prob


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # 接続ごとに独立した landmark buffer
    landmark_buffer = deque(maxlen=T)

    print("✅ Client connected")

    while True:
        try:
            data = await ws.receive_text()
        except:
            print("❌ Client disconnected")
            return

        # data = JSON文字列
        obj = json.loads(data)
        vec = obj["landmark"]  # shape: 225 list

        # landmark buffer に追加
        landmark_buffer.append(vec)

        result = {
            "label": None,
            "prob": 0.0,
            "ready": False
        }

        # T フレーム揃ったら推論
        if len(landmark_buffer) == T:
            label, prob = predict_sequence(landmark_buffer)
            result["label"] = label
            result["prob"] = prob
            result["ready"] = True

        await ws.send_text(json.dumps(result))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
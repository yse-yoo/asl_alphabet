import uvicorn
import os
import json
import numpy as np
import tensorflow as tf
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

# ===============================
# ✅ FastAPI
# ===============================
app = FastAPI()

origins = ["*"]   # WebSocket を含め完全許可

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ✅ 推論関数（T×225 → softmax → label）
# ===============================
def predict_sequence(buffer):
    # shape = (1, T, 225)
    x = np.array(buffer, dtype=np.float32).reshape(1, T, LAND_DIM)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    prob = float(pred[idx])
    label = ASL_CLASSES[idx]
    return label, prob

# ===============================
# ✅ WebSocket
# ===============================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected")

    # WebSocketごとに landmark buffer を独立保持
    buffer = deque(maxlen=T)

    while True:
        try:
            msg = await ws.receive_text()
        except Exception:
            print("❌ Client disconnected")
            break

        try:
            obj = json.loads(msg)
            vec = obj.get("landmark", [])
        except:
            continue

        # landmark をバッファに追加
        if len(vec) == LAND_DIM:
            buffer.append(vec)

        # 返却用データ
        result = {
            "ready": False,
            "label": "...",
            "prob": 0.0
        }

        # Tフレーム揃ったら推論
        if len(buffer) == T:
            label, prob = predict_sequence(buffer)
            result["ready"] = True
            result["label"] = label
            result["prob"] = prob

        print(result)
        # クライアントへ返信
        await ws.send_text(json.dumps(result))


# ===============================
# ✅ 実行
# ===============================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

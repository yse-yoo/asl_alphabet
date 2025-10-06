import cv2
import os
import mediapipe as mp

SAVE_DIR = "asl_finetune_data/nothing"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

count = 0
MAX_COUNT = 100  # ← ここで上限設定

print("📷 nothing クラスの自動撮影開始（手を出さない状態でカメラ前に立ってください）")
print("🛑 Qキーで終了、または100枚保存で自動停止")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        save_path = os.path.join(SAVE_DIR, f"nothing_{count:03d}.jpg")
        cv2.imwrite(save_path, frame)
        count += 1
        cv2.putText(frame, f"Saved {count}/{MAX_COUNT}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Capture nothing", frame)

    # Qキーで中断 or 100枚撮ったら自動終了
    if (cv2.waitKey(1) & 0xFF == ord("q")) or count >= MAX_COUNT:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ nothing クラスを {count} 枚保存しました。")
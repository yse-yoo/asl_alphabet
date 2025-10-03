import cv2
import os
import mediapipe as mp

# 保存先フォルダ
DATA_DIR = "asl_finetune_data"

# クラス一覧（A〜Z + space + nothing）
classes = [chr(i) for i in range(ord("A"), ord("Z")+1)] + ["space", "nothing"]

# 各クラスの保存先フォルダを作成
for cls in classes:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# カメラ起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ カメラが見つかりません")
    exit()

print("✅ カメラ起動成功")
print("👉 数字キー or a〜zキーでクラス切り替え, Sキーで保存, Qキーで終了")

current_class = "A"
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 反転（鏡映し）
    # frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe で手検出
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    bbox = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # バウンディングボックス計算
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            min_x, max_x = int(min(xs)) - 20, int(max(xs)) + 20
            min_y, max_y = int(min(ys)) - 20, int(max(ys)) + 20
            bbox = (min_x, min_y, max_x, max_y)

            # 手の骨格を描画
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # クラス名表示
    cv2.putText(frame, f"Class: {current_class} | Saved: {img_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ASL Hand Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # Qで終了
    if key == ord("q"):
        break

    # Sで保存（手が認識された場合のみ）
    elif key == ord("s") and bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w, max_x), min(h, max_y)
        crop = frame[min_y:max_y, min_x:max_x]

        save_path = os.path.join(DATA_DIR, current_class, f"{current_class}_{img_count:03d}.jpg")
        cv2.imwrite(save_path, crop)
        print(f"💾 保存: {save_path}")
        img_count += 1

    # 数字キーでクラス切り替え（例: 1=A, 2=B ...）
    elif key in range(ord("1"), ord("9")+1):
        idx = key - ord("1")
        if idx < len(classes):
            current_class = classes[idx]
            img_count = len(os.listdir(os.path.join(DATA_DIR, current_class)))
            print(f"🔄 クラス変更: {current_class}")

    # a〜zキーで直接切り替え
    elif key in range(ord("a"), ord("z")+1):
        letter = chr(key).upper()
        if letter in classes:
            current_class = letter
            img_count = len(os.listdir(os.path.join(DATA_DIR, current_class)))
            print(f"🔄 クラス変更: {current_class}")

cap.release()
cv2.destroyAllWindows()

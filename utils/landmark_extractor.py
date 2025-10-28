import os
import cv2
import json
import numpy as np
import mediapipe as mp

# ==============================
# MediaPipe 初期化
# ==============================
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


# ==============================
# ランドマーク抽出
# ==============================
def extract_landmarks_from_image(image, mode="both"):
    """
    画像からPose(33点)＋Hands(最大2x21点)を抽出
    mode: "pose", "hands", "both"
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)

    pose_points = []
    hands_points = []

    # --- Pose 33点 ---
    if pose_results.pose_landmarks and mode in ["pose", "both"]:
        for lm in pose_results.pose_landmarks.landmark:
            pose_points.append({
                "x_norm": lm.x,
                "y_norm": lm.y,
                "z_norm": lm.z,
            })

    # --- Hands 21点×N ---
    if hands_results.multi_hand_landmarks and mode in ["hands", "both"]:
        for hand in hands_results.multi_hand_landmarks:
            hand_data = []
            for lm in hand.landmark:
                hand_data.append({
                    "x_norm": lm.x,
                    "y_norm": lm.y,
                    "z_norm": lm.z,
                })
            hands_points.append(hand_data)

    return {"pose": pose_points, "hands": hands_points}


# ==============================
# 差分補完モード
# ==============================
def update_json_dataset(base_dir, classes, mode="both"):
    """
    既存JSONを保持しつつ、欠損している項目のみ補完する安全モード
    """
    total_updated = 0
    total_created = 0

    for cls in classes:
        folder = os.path.join(base_dir, cls)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(folder, file)
            base_name = os.path.splitext(file)[0]
            json_path = os.path.join(folder, f"{base_name}.json")

            img = cv2.imread(img_path)
            if img is None:
                continue

            new_data = extract_landmarks_from_image(img, mode=mode)

            if os.path.exists(json_path):
                # --- 既存ファイルを読み込み ---
                with open(json_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)

                updated = False

                # --- pose 欠損時のみ補完 ---
                if ("pose" not in old_data or not old_data["pose"]) and new_data["pose"]:
                    old_data["pose"] = new_data["pose"]
                    updated = True

                # --- hands 欠損時のみ補完 ---
                if ("hands" not in old_data or not old_data["hands"]) and new_data["hands"]:
                    old_data["hands"] = new_data["hands"]
                    updated = True

                if updated:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(old_data, f, ensure_ascii=False, indent=2)
                    print(f"🩵 更新: {cls}/{file}")
                    total_updated += 1

            else:
                # --- JSONが無い場合は新規作成 ---
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                print(f"🆕 生成: {cls}/{file}")
                total_created += 1

    print(f"\n📊 差分補完完了: 更新 {total_updated}件 / 新規 {total_created}件")


# ==============================
# 通常生成モード（上書きも可）
# ==============================
def generate_json_dataset(base_dir, classes, output_dir=None, mode="both", overwrite=False):
    """
    各画像からPose+Handsを抽出してJSON保存
    overwrite=True の場合、既存ファイルも上書き
    """
    if output_dir is None:
        output_dir = base_dir

    os.makedirs(output_dir, exist_ok=True)
    total = 0

    for cls in classes:
        folder = os.path.join(base_dir, cls)
        if not os.path.exists(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(folder, file)
            base_name = os.path.splitext(file)[0]
            json_path = os.path.join(folder, f"{base_name}.json")

            if os.path.exists(json_path) and not overwrite:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            data = extract_landmarks_from_image(img, mode=mode)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            total += 1
            print(f"✅ {cls}/{file} → {json_path}")

    print(f"\n📊 合計 {total} ファイルを処理完了")
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils


# ------------------------------------
# 単体の landmark を点だけ描画する関数
# ------------------------------------
def draw_points_only(
    frame,
    landmark_list,
    color=(0, 255, 0),
    thickness=4,
    radius=3
):
    """
    MediaPipe の LandmarkList を点のみで描画する。
    （線なし connections=None）
    """
    if landmark_list is None:
        return

    mp_draw.draw_landmarks(
        frame,
        landmark_list,
        connections=None,  # ✅ ラインなし
        landmark_drawing_spec=mp_draw.DrawingSpec(
            color=color,
            thickness=thickness,
            circle_radius=radius
        ),
    )


# ------------------------------------
# Pose + Hands をまとめて描画する関数
# ------------------------------------
def draw_skeleton_points(
    frame,
    pose_result,
    hands_result,
    pose_color=(0, 255, 0),
    hand_color=(0, 0, 255),
    thickness=4,
    radius=3
):
    # Pose 点
    if pose_result and pose_result.pose_landmarks:
        draw_points_only(
            frame,
            pose_result.pose_landmarks,
            color=pose_color,
            thickness=thickness,
            radius=radius
        )

    # Hand 点
    if hands_result and hands_result.multi_hand_landmarks:
        for hl in hands_result.multi_hand_landmarks:
            draw_points_only(
                frame,
                hl,
                color=hand_color,
                thickness=thickness,
                radius=radius
            )

ASL_CLASSES = [
    "Hello",
    "I_Love_You",
    "Nothing",
    "Thank_You",
]

# 表示用マッピング
LABEL_MAP = {
    "Hello": "Hello",
    "I_Love_You": "I Love You",
    "Nothing": "Nothing",
    "Thank_You": "Thank You",
}

HAND_WEIGHT = 2.0  # 手のランドマークの重み付け

T = 40                # 時系列長（学習と合わせる）
LAND_DIM = 225        # 1フレームの次元（pose+hands最大225）
# 予測の安定化
PRED_SMOOTH = 5       # 直近N回の予測を平均
PROB_THRESH = 0.50    # この確率未満は「…」表示
START_MOV_THRESH = 0.012    # これ以上の動きが START_FRAMES 続けば「動作中」へ
STOP_MOV_THRESH  = 0.007    # これ未満の動きが STOP_FRAMES 続けば「静止」へ
START_FRAMES = 3            # 動作開始に必要な連続フレーム数
STOP_FRAMES  = 8            # 動作停止に必要な連続フレーム数


USE_MODEL="asl_multimodal_model_hands"
ALPHABET_DIR = "asl_alphabet_train"
VIDEO_DIR = "asl_videos"
DATA_DIR = "asl_words_train"
MODEL_DIR = "models"
TEST_DIR ="asl_words_test"
EXTENTION = "keras"
IMAGE_SIZE = (64, 64)
MARGIN = 100

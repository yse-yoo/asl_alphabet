from asl_config import ASL_CLASSES, DATA_DIR
from utils.landmark_extractor import update_json_dataset

# 欠損だけ補う安全モード
update_json_dataset(DATA_DIR, ASL_CLASSES, mode="both")
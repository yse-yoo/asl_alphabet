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

ALPHABET_DIR = "asl_alphabet_train"
DATA_DIR = "asl_words_train"
MODEL_DIR = "models"
TEST_DIR ="asl_words_test"
EXTENTION = "keras"
IMAGE_SIZE = (64, 64)
MARGIN = 100

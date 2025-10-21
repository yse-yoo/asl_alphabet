from asl_config import LABEL_MAP

def normalize_label(label: str) -> str:
    """
    モデル出力ラベルを人間が読みやすい形に変換。
    未定義ラベルは自動的に "_" を " " に変換。
    """
    return LABEL_MAP.get(label, label.replace("_", " "))

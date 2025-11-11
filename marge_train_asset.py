import os
import shutil
import re

def merge_and_move(src_dir, dst_dir):
    """
    src_dir の ASL データを dst_dir にマージする。
    ファイルはコピーではなく「リネームして移動」する。
    重複名は連番を振り直して回避する。
    """

    # src_dir 内のクラスフォルダを走査
    for cls in os.listdir(src_dir):
        src_cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls_path):
            continue

        # dst のクラスフォルダを作成（なければ）
        dst_cls_path = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_path, exist_ok=True)

        # 既存ファイルから最大番号を取得
        existing_files = [f for f in os.listdir(dst_cls_path) if f.startswith(cls)]
        pattern = re.compile(rf"{cls}_(\d+)\.json")

        max_index = -1
        for fname in existing_files:
            m = pattern.match(fname)
            if m:
                n = int(m.group(1))
                max_index = max(max_index, n)

        # src 内のファイルを順番に処理
        for fname in os.listdir(src_cls_path):
            src_file = os.path.join(src_cls_path, fname)

            # JSON形式の連番ファイルかどうか判定
            m = pattern.match(fname)
            if m:
                # 新しい番号で保存
                max_index += 1
                new_name = f"{cls}_{max_index:03d}.json"
            else:
                # もし違う形式のファイルの場合は適当に末尾番号
                base, ext = os.path.splitext(fname)
                max_index += 1
                new_name = f"{base}_{max_index}{ext}"

            dst_file = os.path.join(dst_cls_path, new_name)

            # ファイルを移動（上書きなし）
            shutil.move(src_file, dst_file)
            print(f"Moved: {src_file} -> {dst_file}")

    print("✅ Merge & Move complete!")


if __name__ == "__main__":
    SOURCE = "asl_videos_from_other_pc"
    DEST = "asl_videos"

    merge_and_move(SOURCE, DEST)

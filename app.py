from __future__ import annotations

import os
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from caption_model import caption_image
from translate_model import translate_en_to_ja

app = Flask(__name__)

# --- 設定 ---
UPLOAD_DIR = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# templatesフォルダからCSS/JSを配信（あなたの希望）
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("templates", filename)


def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    image_path = ""          # 例: static/uploads/xxx.jpg
    english_text = ""
    japanese_text = ""
    error_message = ""
    show_preview = False

    if request.method == "POST":
        show_preview = True

        file = request.files.get("image_file")
        if not file or file.filename == "":
            error_message = "画像ファイルが選択されていません。"
        else:
            filename = secure_filename(file.filename)
            if not _allowed_file(filename):
                error_message = f"対応していない拡張子です（{', '.join(sorted(ALLOWED_EXTENSIONS))}）"
            else:
                # 同名衝突回避のため、タイムスタンプを付与
                stem, ext = os.path.splitext(filename)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                saved_name = f"{stem}_{ts}{ext}"
                save_path = os.path.join(UPLOAD_DIR, saved_name)

                try:
                    file.save(save_path)
                    image_path = f"static/uploads/{saved_name}"

                    # 既存関数をそのまま利用
                    english_text = caption_image(image_path)
                    japanese_text = translate_en_to_ja(english_text)

                except Exception as e:
                    error_message = str(e)

    return render_template(
        "index.html",
        image_path=image_path,
        english_text=english_text,
        japanese_text=japanese_text,
        error_message=error_message,
        show_preview=show_preview,
    )


if __name__ == "__main__":
    app.run(debug=True)

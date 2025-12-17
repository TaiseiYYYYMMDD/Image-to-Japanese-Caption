#from translate_model_nllb import translate_en_to_ja
from flask import Flask, render_template, request, send_from_directory
from caption_model import caption_image
from translate_model import translate_en_to_ja

app = Flask(__name__)

# ★追加：templatesフォルダからCSS/JSを配信
@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("templates", filename)

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = ""
    english_text = ""
    japanese_text = ""
    error_message = ""

    if request.method == "POST":
        image_path = request.form.get("image_path", "").strip(' "\'') or image_path
        try:
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
    )

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
from caption_model import caption_image
#from translate_model import translate_en_to_ja
from translate_model_nllb import translate_en_to_ja


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = "static/uploads/dummy.jpg"
    english_text = ""
    japanese_text = ""

    if request.method == "POST":
        # HTMLフォームから受け取る
        image_path = request.form.get("image_path", "").strip() or image_path

        # 既存の関数をそのまま呼ぶ（保持）
        english_text = caption_image(image_path)
        japanese_text = translate_en_to_ja(english_text)

    return render_template(
        "index.html",
        image_path=image_path,
        english_text=english_text,
        japanese_text=japanese_text,
    )


if __name__ == "__main__":
    app.run(debug=True)

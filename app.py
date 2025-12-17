from flask import Flask, render_template
from caption_model import caption_image
from translate_model import translate_en_to_ja

app = Flask(__name__)

@app.route("/")
def index():
    # 仮：後で image_path はアップロードされた画像に置き換える
    image_path = "static/uploads/dummy.jpg"
    english = caption_image(image_path)
    japanese = translate_en_to_ja(english)
    return render_template("index.html", english_text=english, japanese_text=japanese)

if __name__ == "__main__":
    app.run(debug=True)

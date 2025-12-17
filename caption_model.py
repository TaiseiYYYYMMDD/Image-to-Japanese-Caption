#caption_image(image_path: str) -> str
# caption_model.py

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# モデル名（課題指定）
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

# モデルと前処理をグローバルでロード（毎回ロードしない）
device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def caption_image(image_path: str) -> str:
    """
    画像を入力として受け取り、
    その画像が何をしているかを英語で説明する
    """

    # 画像を読み込み（RGBに変換）
    image = Image.open(image_path).convert("RGB")

    # 前処理
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # キャプション生成
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=30,
            num_beams=4
        )

    # トークン → 文字列
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

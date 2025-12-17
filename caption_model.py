# caption_model.py
from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# 追加：評価用
import evaluate

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()


def caption_image(image_path: str) -> str:
    """
    画像を入力として受け取り、その画像が何をしているかを英語で説明する
    """
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=30,
            num_beams=4
        )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def evaluate_caption_model(
    samples: List[Dict[str, Any]],
    max_length: int = 30,
    num_beams: int = 4,
) -> Dict[str, Any]:
    """
    キャプションモデルの性能を評価する。

    samples の形式（例）:
    [
      {"image_path": "path/to/a.jpg", "references": ["a dog playing with a ball", "a dog plays with a ball"]},
      {"image_path": "path/to/b.jpg", "references": ["a man riding a bicycle"]},
    ]

    返り値:
      - BLEU, METEOR, ROUGE-L, CIDEr（利用できるもの）
      - 平均推論時間（秒）
      - 予測結果一覧（必要なら）
    """

    # 利用する指標（入っていない環境もあるので安全に）
    metrics = {}
    # sacrebleu
    try:
        metrics["bleu"] = evaluate.load("sacrebleu")
    except Exception:
        pass
    # meteor
    try:
        metrics["meteor"] = evaluate.load("meteor")
    except Exception:
        pass
    # rouge（ROUGE-Lを使う）
    try:
        metrics["rouge"] = evaluate.load("rouge")
    except Exception:
        pass
    # cider
    try:
        metrics["cider"] = evaluate.load("cider")
    except Exception:
        pass
    # spice（環境によって入らないことが多い）
    try:
        metrics["spice"] = evaluate.load("spice")
    except Exception:
        pass

    predictions: List[str] = []
    references_for_metrics: Dict[str, Any] = {
        "refs_single": [],     # meteor/rouge等で使える形（1つに絞る）
        "refs_multi": [],      # cider/spiceで使う（複数参照）
    }

    total_time = 0.0

    for s in samples:
        image_path = s["image_path"]
        refs: List[str] = s.get("references", [])
        if not refs:
            raise ValueError(f"references が空です: {image_path}")

        # 推論（速度計測込み）
        start = time.perf_counter()

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams
            )

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        end = time.perf_counter()
        total_time += (end - start)

        predictions.append(pred)

        # 指標用に整形
        # 多くの指標は参照を1本にするだけでも計算可能（最小）
        references_for_metrics["refs_single"].append(refs[0])
        # CIDEr/SPICEは参照が複数あるほど妥当
        references_for_metrics["refs_multi"].append(refs)

    avg_latency_sec = total_time / max(len(samples), 1)

    results: Dict[str, Any] = {
        "num_samples": len(samples),
        "avg_latency_sec": avg_latency_sec,
    }

    # BLEU: refs は list[list[str]] 形式が必要（各予測に対して参照リスト）
    if "bleu" in metrics:
        bleu_refs = [[r] for r in references_for_metrics["refs_single"]]
        results["bleu"] = metrics["bleu"].compute(predictions=predictions, references=bleu_refs)

    # METEOR: refs は list[str] でOK
    if "meteor" in metrics:
        results["meteor"] = metrics["meteor"].compute(
            predictions=predictions,
            references=references_for_metrics["refs_single"]
        )

    # ROUGE: まとめて出るのでROUGE-Lだけ読む
    if "rouge" in metrics:
        rouge_out = metrics["rouge"].compute(
            predictions=predictions,
            references=references_for_metrics["refs_single"]
        )
        results["rouge"] = {
            "rougeL": rouge_out.get("rougeL", None),
            "rouge1": rouge_out.get("rouge1", None),
            "rouge2": rouge_out.get("rouge2", None),
        }

    # CIDEr: refs は list[list[str]]
    if "cider" in metrics:
        results["cider"] = metrics["cider"].compute(
            predictions=predictions,
            references=references_for_metrics["refs_multi"]
        )

    # SPICE: refs は list[list[str]]
    if "spice" in metrics:
        results["spice"] = metrics["spice"].compute(
            predictions=predictions,
            references=references_for_metrics["refs_multi"]
        )

    # 必要なら出力例も返す（レポート用）
    results["examples"] = [
        {"image_path": samples[i]["image_path"], "prediction": predictions[i], "reference": references_for_metrics["refs_single"][i]}
        for i in range(min(len(samples), 5))
    ]

    return results

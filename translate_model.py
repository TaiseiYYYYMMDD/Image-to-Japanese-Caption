#translate_en_to_ja(text: str) -> str

# def translate_en_to_ja(text: str) -> str:
#     return "ボールで遊んでいる犬"

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslateModel:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-jap") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate_en_to_ja(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=96,
            num_beams=4,
            early_stopping=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ★ これを追加（最重要）
_translator = TranslateModel()

def translate_en_to_ja(text: str) -> str:
    return _translator.translate_en_to_ja(text)

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "jpn_Jpan"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
_model.eval()

@torch.no_grad()
def translate_en_to_ja(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    _tokenizer.src_lang = SRC_LANG
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(_device)

    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(TGT_LANG)

    outputs = _model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=4,
        early_stopping=True,
    )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

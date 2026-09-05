import json
from typing import Optional

import torch
from pathlib import Path
from unicodedata import normalize
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from Zoo.Chatterbox.utils.language_normalization import ChineseCangjieConverter, _japanese_normalize, _korean_normalize

SOT, EOT, UNK, SPACE = "[START]", "[STOP]", "[UNK]", "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]


class MTLTokenizer:
    def __init__(self, vocab_path: str, cj_path: Optional[str] = None):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_path)
        voc = self.tokenizer.get_vocab()
        assert SOT in voc and EOT in voc, "Vocabulary missing [START] or [STOP] tokens."
        self.cj_converter = ChineseCangjieConverter(cj_path) if cj_path and Path(cj_path).exists() else None

    @classmethod
    def from_pretrained(cls, repo_id: str = "ResembleAI/chatterbox", cache_dir: Optional[str] = None):
        """Loads vocabulary & Cangjie mapping directly from Hugging Face."""
        vocab_file = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=cache_dir)
        cj_file = hf_hub_download(repo_id=repo_id, filename="Cangjie5_TC.json", cache_dir=cache_dir)
        return cls(vocab_path=vocab_file, cj_path=cj_file)

    def _normalize(self, text: str, lang: Optional[str] = None) -> str:
        text = normalize("NFKD", text.lower())
        
        # Language-specific phonetic/structural routing
        if lang == "zh" and self.cj_converter:
            text = self.cj_converter(text)
        elif lang == "ja":
            text = _japanese_normalize(text)
        elif lang == "ko":
            text = _korean_normalize(text)
        elif lang == "he":
            try:
                from dicta_onnx import Dicta
                text = Dicta(model_path="").add_diacritics(text)
            except ImportError: pass
        elif lang == "ru":
            # Skip for Now Since there is problem with the Russian normalization library
            pass

        if lang and lang != "en":
            text = f"[{lang.lower()}]{text}"
            
        return text.replace(" ", SPACE)

    def text_to_tokens(self, text: str, lang: Optional[str] = None, device: str = "cpu") -> torch.Tensor:
        """Converts raw text to model input tensor [1, Seq_Len]."""
        clean_text = self._normalize(text, lang=lang)
        token_ids = self.tokenizer.encode(clean_text).ids
        return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    def decode(self, seq) -> str:
        """Converts token sequence back to readable text."""
        if isinstance(seq, torch.Tensor):
            seq = seq.squeeze().cpu().tolist()
        text = self.tokenizer.decode(seq, skip_special_tokens=False)
        for token in [EOT, UNK, " "]:
            text = text.replace(token, "")
        return text.replace(SPACE, " ").strip()




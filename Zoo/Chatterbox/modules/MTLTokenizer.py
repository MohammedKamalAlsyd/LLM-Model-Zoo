"""Multilingual BPE Tokenizer for Chatterbox supporting Cangjie and CJK normalization."""

from pathlib import Path
from typing import Optional, Union
from unicodedata import normalize
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from Zoo.Chatterbox.processing.language_normalization import (
    ChineseCangjieConverter,
    japanese_normalize,
    korean_normalize,
)

SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]


class MTLTokenizer:
    """Pre-trained Hugging Face Byte-Pair Encoding tokenizer adapted for 23 languages."""

    def __init__(self, vocab_path: Union[str, Path], cj_path: Optional[Union[str, Path]] = None) -> None:
        self.tokenizer: Tokenizer = Tokenizer.from_file(str(vocab_path))
        voc = self.tokenizer.get_vocab()
        assert SOT in voc and EOT in voc, "Vocabulary missing [START] or [STOP] token identifiers."
        self.cj_converter = (
            ChineseCangjieConverter(cj_path) if cj_path and Path(cj_path).exists() else None
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "ResembleAI/chatterbox",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "MTLTokenizer":
        """Downloads tokenizer and Cangjie definitions directly from Hugging Face Hub."""
        vocab_file = hf_hub_download(
            repo_id=repo_id,
            filename="grapheme_mtl_merged_expanded_v1.json",
            cache_dir=cache_dir,
        )
        cj_file = hf_hub_download(
            repo_id=repo_id,
            filename="Cangjie5_TC.json",
            cache_dir=cache_dir,
        )
        return cls(vocab_path=vocab_file, cj_path=cj_file)

    def normalize(self, text: str, lang: Optional[str] = None) -> str:
        """Applies Unicode decomposition, CJK parsing, and language prefixing."""
        text = normalize("NFKD", text.lower())

        if lang == "zh" and self.cj_converter:
            text = self.cj_converter(text)
        elif lang == "ja":
            text = japanese_normalize(text)
        elif lang == "ko":
            text = korean_normalize(text)
        elif lang == "he":
            try:
                from dicta_onnx import Dicta
                text = Dicta(model_path="").add_diacritics(text)
            except ImportError:
                pass

        if lang and lang.lower() != "en":
            text = f"[{lang.lower()}]{text}"

        return text.replace(" ", SPACE)

    def text_to_tokens(
        self,
        text: str,
        lang: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Encodes string input into a token tensor of shape (1, seq_len)."""
        clean_text = self.normalize(text, lang=lang)
        token_ids = self.tokenizer.encode(clean_text).ids
        return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    def decode(self, seq: Union[torch.Tensor, list]) -> str:
        """Decodes token ID sequence back into UTF-8 text."""
        if isinstance(seq, torch.Tensor):
            seq = seq.squeeze().cpu().tolist()
        text = self.tokenizer.decode(seq, skip_special_tokens=False)
        for token in [EOT, UNK, " "]:
            text = text.replace(token, "")
        return text.replace(SPACE, " ").strip()
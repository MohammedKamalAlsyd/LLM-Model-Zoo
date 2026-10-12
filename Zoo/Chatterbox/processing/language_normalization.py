"""High-precision phonetic and structural language normalization for multilingual speech synthesis."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, cast
from unicodedata import category, normalize


def korean_normalize(text: str) -> str:
    """Decomposes Korean Hangul syllables into standard Unicode Jamo components.

    Args:
        text: Raw Korean input string.

    Returns:
        Decomposed Jamo string.
    """
    res: List[str] = []
    for c in text:
        code = ord(c)
        if 0xAC00 <= code <= 0xD7AF:
            base = code - 0xAC00
            initial = chr(0x1100 + base // 588)
            medial = chr(0x1161 + (base % 588) // 28)
            res.extend([initial, medial])
            final_idx = base % 28
            if final_idx > 0:
                res.append(chr(0x11A7 + final_idx))
        else:
            res.append(c)
    return "".join(res).strip()


def japanese_normalize(text: str) -> str:
    """Converts Kanji to Hiragana and applies NFKD normalization.

    Args:
        text: Raw Japanese text.

    Returns:
        Phonetically expanded Hiragana string.
    """
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        res: List[str] = []
        for r in kks.convert(text):
            orig, hira = r["orig"], r["hira"]
            if any(19968 <= ord(c) <= 40959 for c in orig):
                res.append((" " if hira.startswith(("は", "へ")) else "") + hira)
            else:
                res.append(orig)
        return normalize("NFKD", "".join(res))
    except ImportError:
        return text


class ChineseCangjieConverter:
    """Decomposes Chinese Hanzi characters into structural Cangjie codes for cross-lingual synthesis."""

    def __init__(self, cj_file_path: Union[str, Path]) -> None:
        self.word2cj: Dict[str, str] = {}
        self.cj2word: Dict[str, List[str]] = {}

        if cj_file_path and Path(cj_file_path).exists():
            with open(cj_file_path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    parts = entry.split("\t")
                    if len(parts) >= 2:
                        w, c = parts[0], parts[1]
                        self.word2cj[w] = c
                        self.cj2word.setdefault(c, []).append(w)
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            self.segmenter = None

    def __call__(self, text: str) -> str:
        words: List[str] = (
            cast(List[str], self.segmenter.cut(text)) if self.segmenter else [text]
        )
        full_text = " ".join(words) if self.segmenter else text
        out: List[str] = []
        for ch in full_text:
            if category(ch) == "Lo" and ch in self.word2cj:
                code = self.word2cj[ch]
                idx = self.cj2word[code].index(ch)
                cj_str = code + (str(idx) if idx > 0 else "")
                out.append("".join(f"[cj_{c}]" for c in cj_str) + "[cj_.]")
            else:
                out.append(ch)
        return "".join(out)


def normalize_punctuation(text: str) -> str:
    """Cleans up punctuation, collapses duplicate spaces, and guarantees sentence closure."""
    if not text or len(text.strip()) == 0:
        return "You need to add some text for me to talk."

    # Capitalize initial character
    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    replacements = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    text = text.rstrip(" ")
    sentence_closers = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_closers):
        text += "."

    return text
import json
from typing import cast
from unicodedata import category, normalize

def _korean_normalize(text: str) -> str:
    """Decomposes Korean Hangul into Jamo components."""
    res = []
    for c in text:
        if 0xAC00 <= ord(c) <= 0xD7AF:
            base = ord(c) - 0xAC00
            res.extend([chr(0x1100 + base // 588), chr(0x1161 + (base % 588) // 28)])
            if base % 28:
                res.append(chr(0x11A7 + base % 28))
        else:
            res.append(c)
    return "".join(res).strip()


def _japanese_normalize(text: str) -> str:
    """Converts Kanji to Hiragana and applies NFKD normalization."""
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        res = []
        for r in kks.convert(text):
            orig, hira = r['orig'], r['hira']
            if any(19968 <= ord(c) <= 40959 for c in orig):
                res.append((" " if hira.startswith(("は", "へ")) else "") + hira)
            else:
                res.append(orig)
        return normalize("NFKD", "".join(res))
    except ImportError:
        return text
    
    
class ChineseCangjieConverter:
    """Decomposes Chinese characters into Cangjie structural codes."""
    def __init__(self, cj_file_path: str):
        self.word2cj, self.cj2word = {}, {}
        with open(cj_file_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                w, c = entry.split("\t")[:2]
                self.word2cj[w] = c
                self.cj2word.setdefault(c, []).append(w)
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            self.segmenter = None

    def __call__(self, text: str) -> str:
        words: list[str] = cast(list[str], self.segmenter.cut(text)) if self.segmenter else [text]
        full_text = " ".join(words) if self.segmenter else text
        out = []
        for ch in full_text:
            if category(ch) == "Lo" and ch in self.word2cj:
                code = self.word2cj[ch]
                idx = self.cj2word[code].index(ch)
                cj_str = code + (str(idx) if idx > 0 else "")
                out.append("".join(f"[cj_{c}]" for c in cj_str) + "[cj_.]")
            else:
                out.append(ch)
        return "".join(out)
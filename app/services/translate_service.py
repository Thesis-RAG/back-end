"""Vietnamese → English translation for auto-filling entity_key from display_name.

Uses deep-translator's free Google Translate endpoint. Best-effort: on any
failure (network, rate limit, package missing) falls back to a diacritics-
stripped version of the original text — never the raw Vietnamese, which
policy_rule_service.normalize_entity_key would otherwise mangle (every
accented char/space run collapses to a single "_", eg. "số điện thoại" ->
"s_i_n_tho_i" instead of a readable "so_dien_thoai").
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache


def _strip_diacritics(text: str) -> str:
    # NFD splits accented Latin letters into base + combining marks, which
    # we then drop — but Vietnamese "đ/Đ" doesn't decompose that way (it's
    # its own base letter), so it needs an explicit swap first.
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


@lru_cache(maxsize=512)
def translate_vi_to_en(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="vi", target="en").translate(text)
        return translated or _strip_diacritics(text)
    except Exception:
        return _strip_diacritics(text)

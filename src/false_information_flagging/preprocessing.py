from __future__ import annotations

import html
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
BRACKET_PATTERN = re.compile(r"\[[^]]*\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html: bool = True,
    strip_urls: bool = True,
    remove_bracketed_text: bool = True,
    remove_stopwords: bool = True,
    remove_punctuation: bool = False,
) -> str:
    value = html.unescape(str(text))

    if strip_html:
        value = HTML_PATTERN.sub(" ", value)
    if strip_urls:
        value = URL_PATTERN.sub(" ", value)
    if remove_bracketed_text:
        value = BRACKET_PATTERN.sub(" ", value)
    if lowercase:
        value = value.lower()
    if remove_punctuation:
        value = value.translate(PUNCT_TRANSLATION_TABLE)

    tokens = value.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

    value = " ".join(tokens)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


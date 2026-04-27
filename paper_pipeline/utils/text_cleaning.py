"""Text normalization and 10-K HTML extraction helpers."""

from __future__ import annotations

import re
from html import unescape

from lxml import etree, html


WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
PAGE_MARKER_RE = re.compile(r"^-?\s*\d+\s*-?$")

PREFERRED_BLOCK_XPATH = (
    "//p"
    " | //li"
    " | //td[not(.//p or .//div or .//li)]"
    " | //th[not(.//p or .//div or .//li)]"
    " | //tr[not(.//td or .//th)]"
    " | //h1 | //h2 | //h3 | //h4 | //h5 | //h6"
)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = unescape(str(text))
    cleaned = cleaned.replace("\x00", " ").replace("\u00a0", " ")
    cleaned = ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_plain_text(text: str | None) -> str:
    return normalize_text(text)


def extract_html_paragraphs(raw_html: str | None) -> list[str]:
    normalized = normalize_text(raw_html)
    if not normalized:
        return []

    try:
        document = html.fromstring(raw_html)
    except (etree.ParserError, ValueError):
        fallback = normalize_text(re.sub(r"<[^>]+>", " ", normalized))
        return [fallback] if fallback else []

    etree.strip_elements(document, "script", "style", "noscript", with_tail=False)

    paragraphs: list[str] = []
    previous = None
    for node in document.xpath(PREFERRED_BLOCK_XPATH):
        text = normalize_text(" ".join(fragment.strip() for fragment in node.itertext()))
        if not text:
            continue
        if PAGE_MARKER_RE.fullmatch(text):
            continue
        if text == previous:
            continue
        paragraphs.append(text)
        previous = text

    if paragraphs:
        return paragraphs

    fallback = normalize_text(" ".join(fragment.strip() for fragment in document.itertext()))
    return [fallback] if fallback else []


def join_paragraphs(paragraphs: list[str]) -> str:
    cleaned = [normalize_text(paragraph) for paragraph in paragraphs if normalize_text(paragraph)]
    return "\n\n".join(cleaned)

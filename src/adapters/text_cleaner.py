"""TechCompass — 文本清洗工具。

用于清洗 Reddit RSS 等来源中的 HTML 片段，避免 HTML 标签污染 RAG 向量与 LLM 证据。
"""

from __future__ import annotations

import html
import re

_BLOCK_TAGS = re.compile(r"</?(?:div|p|br|ul|ol|li|table|tr|td|th|h[1-6])\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """移除 HTML 标签并规范化空白。"""
    if not text:
        return ""
    text = _BLOCK_TAGS.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_html_like(text: str) -> bool:
    """粗略判断文本是否包含 HTML。"""
    return bool(text and re.search(r"<(?:div|p|a|br|span|ul|ol|li|table|tr|td|th)\b", text, re.IGNORECASE))

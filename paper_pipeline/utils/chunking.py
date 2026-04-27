"""Tokenizer-aware chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    token_count: int
    chunk_start_token: int
    chunk_end_token: int


def count_tokens_from_ids(token_ids: list[int], special_tokens: int) -> int:
    return len(token_ids) + special_tokens


def token_chunk_from_ids(
    token_ids: list[int],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
    special_tokens: int,
    start_offset: int = 0,
) -> list[Chunk]:
    if max_tokens <= special_tokens:
        raise ValueError("max_tokens must be greater than the number of special tokens.")

    max_content_tokens = max_tokens - special_tokens
    if max_content_tokens <= 0:
        raise ValueError("No room left for content tokens after accounting for special tokens.")

    if not token_ids:
        return []

    if len(token_ids) <= max_content_tokens:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        return [
            Chunk(
                text=decoded,
                token_count=len(token_ids) + special_tokens,
                chunk_start_token=start_offset,
                chunk_end_token=start_offset + len(token_ids) - 1,
            )
        ]

    stride = max_content_tokens - overlap_tokens
    if stride <= 0:
        raise ValueError("overlap_tokens must be smaller than the content token budget.")

    chunks: list[Chunk] = []
    cursor = 0
    while cursor < len(token_ids):
        window = token_ids[cursor : cursor + max_content_tokens]
        if not window:
            break
        decoded = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        chunks.append(
            Chunk(
                text=decoded,
                token_count=len(window) + special_tokens,
                chunk_start_token=start_offset + cursor,
                chunk_end_token=start_offset + cursor + len(window) - 1,
            )
        )
        if cursor + max_content_tokens >= len(token_ids):
            break
        cursor += stride

    return chunks

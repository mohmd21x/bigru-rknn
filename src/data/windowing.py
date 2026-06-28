"""Sliding-window utilities for fixed-length temporal sequences."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.constants import DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STRIDE


@dataclass(frozen=True)
class WindowSlice:
    """One window over a clip sequence."""

    start: int
    end: int
    top_kp: np.ndarray
    bot_kp: np.ndarray
    feat: np.ndarray
    mask: np.ndarray


def _pad_sequence(array: np.ndarray, target_length: int) -> np.ndarray:
    """Zero-pad a ``(T, D)`` array along time to ``target_length``."""
    length = array.shape[0]
    if length >= target_length:
        return array[:target_length]
    pad_width = [(0, target_length - length)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_width, mode="constant", constant_values=0.0)


def _pad_mask(mask: np.ndarray, target_length: int) -> np.ndarray:
    """Pad a ``(T,)`` validity mask with zeros."""
    length = mask.shape[0]
    if length >= target_length:
        return mask[:target_length]
    return np.pad(mask, (0, target_length - length), mode="constant", constant_values=0.0)


def _extract_window(
    top_kp: np.ndarray,
    bot_kp: np.ndarray,
    feat: np.ndarray,
    valid_mask: np.ndarray,
    start: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice ``[start:start+window_size)`` and pad at the end when needed."""
    end = start + window_size
    window_top = _pad_sequence(top_kp[start:end], window_size)
    window_bot = _pad_sequence(bot_kp[start:end], window_size)
    window_feat = _pad_sequence(feat[start:end], window_size)
    window_mask = _pad_mask(valid_mask[start:end], window_size)
    return window_top, window_bot, window_feat, window_mask


def slide_windows(
    top_kp: np.ndarray,
    bot_kp: np.ndarray,
    feat: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> Iterator[WindowSlice]:
    """Yield sliding windows over aligned clip tensors.

    Short clips (``T < window_size``) produce a single zero-padded window.
    Longer clips emit windows every ``stride`` frames; the final window is
    included when the last stride position does not already cover the tail.
    """
    if top_kp.shape[0] != bot_kp.shape[0] or top_kp.shape[0] != feat.shape[0]:
        raise ValueError("top_kp, bot_kp, and feat must have the same length")
    if valid_mask.shape[0] != top_kp.shape[0]:
        raise ValueError("valid_mask length must match sequence length")

    seq_len = top_kp.shape[0]
    if seq_len == 0:
        return

    if seq_len <= window_size:
        window_top, window_bot, window_feat, window_mask = _extract_window(
            top_kp, bot_kp, feat, valid_mask, start=0, window_size=window_size
        )
        yield WindowSlice(
            start=0,
            end=seq_len,
            top_kp=window_top,
            bot_kp=window_bot,
            feat=window_feat,
            mask=window_mask,
        )
        return

    starts = list(range(0, seq_len - window_size + 1, stride))
    last_start = seq_len - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = min(start + window_size, seq_len)
        window_top, window_bot, window_feat, window_mask = _extract_window(
            top_kp, bot_kp, feat, valid_mask, start=start, window_size=window_size
        )
        yield WindowSlice(
            start=start,
            end=end,
            top_kp=window_top,
            bot_kp=window_bot,
            feat=window_feat,
            mask=window_mask,
        )


def valid_frame_ratio(mask: np.ndarray) -> float:
    """Fraction of frames marked valid in a window mask."""
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0))

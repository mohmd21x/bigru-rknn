"""Rolling window buffer for realtime fall classification."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.constants import BOTTOM_KPT_DIM, DEFAULT_WINDOW_SIZE, FEAT_DIM, TOP_KPT_DIM


class FallFrameBuffer:
    """Maintain the latest fixed-length window of pose features.

    ``clip_frames`` controls how many real frames are buffered before the
    model is invoked.  When ``clip_frames < window_size`` the batch returned
    by :meth:`as_batch` is zero-padded at the **end** to ``window_size``,
    matching the training-time padding produced by
    :func:`src.data.windowing.slide_windows` for clips shorter than the
    window.  This is important for models trained on short clips (e.g. the
    7.5 FPS model where each clip yields only ~14 frames inside a window of
    32): leaving ``clip_frames`` at the default (``window_size``) causes the
    model to see 32 real frames at every prediction step, whereas training
    always supplied 14 real frames followed by 18 zeros.

    Args:
        window_size: Sequence length expected by the model (tensor axis T).
        clip_frames: Number of real frames to accumulate before predicting.
            Defaults to ``window_size`` (original behaviour – no padding).
            Set this to the typical training-clip length (e.g. 14 for the
            7.5 FPS model) so that inference windows match the training
            distribution.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        clip_frames: int | None = None,
    ) -> None:
        self.window_size = window_size
        # clip_frames must not exceed window_size.
        self.clip_frames = min(clip_frames, window_size) if clip_frames is not None else window_size
        self._frames: deque[dict[str, Any]] = deque(maxlen=self.clip_frames)

    def reset(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, frame: dict[str, Any]) -> None:
        self._frames.append(frame)

    @property
    def is_ready(self) -> bool:
        return len(self._frames) == self.clip_frames

    def valid_ratio(self) -> float:
        """Fraction of valid frames, denominator is ``window_size``.

        Using ``window_size`` (not ``clip_frames``) as the denominator keeps
        the ratio consistent with the ``min_valid_frame_ratio`` threshold
        that was used during training (where short clips are padded to
        ``window_size`` before that check is applied).
        """
        if not self._frames:
            return 0.0
        valid = sum(float(frame["mask"]) for frame in self._frames)
        return valid / self.window_size

    def as_batch(self) -> dict[str, np.ndarray]:
        """Return numpy arrays shaped ``(1, window_size, D)`` for the model.

        If ``clip_frames < window_size``, the sequence is zero-padded at the
        end to ``window_size``, replicating the training-time padding.
        """
        if not self.is_ready:
            raise RuntimeError("Window buffer is not full yet")

        top = np.stack([frame["top_kp"] for frame in self._frames], axis=0)
        bot = np.stack([frame["bot_kp"] for frame in self._frames], axis=0)
        feat = np.stack([frame["feat"] for frame in self._frames], axis=0)
        mask = np.array([frame["mask"] for frame in self._frames], dtype=np.float32)

        if self.clip_frames < self.window_size:
            pad = self.window_size - self.clip_frames
            top = np.concatenate([top, np.zeros((pad, TOP_KPT_DIM), dtype=np.float32)], axis=0)
            bot = np.concatenate([bot, np.zeros((pad, BOTTOM_KPT_DIM), dtype=np.float32)], axis=0)
            feat = np.concatenate([feat, np.zeros((pad, FEAT_DIM), dtype=np.float32)], axis=0)
            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)], axis=0)

        return {
            "top_kp": top.reshape(1, self.window_size, TOP_KPT_DIM),
            "bot_kp": bot.reshape(1, self.window_size, BOTTOM_KPT_DIM),
            "feat": feat.reshape(1, self.window_size, FEAT_DIM),
            "mask": mask.reshape(1, self.window_size),
        }

#!/usr/bin/env python3
"""RTMO-S ONNX inference CLI — thin wrapper around src.inference.rtmo_pose."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.rtmo_pose import (  # noqa: E402
    DEFAULT_ONNX,
    SCORE_THR,
    create_ort_session,
    draw_poses,
    infer_rtmo_s,
    resolve_rtmo_onnx,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="RTMO-S ONNX pose inference (standalone test).")
    ap.add_argument("image_or_video")
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--score-thr", type=float, default=SCORE_THR)
    ap.add_argument("--out", default="rtmo_s_onnx_out.jpg")
    args = ap.parse_args()

    onnx_path = resolve_rtmo_onnx(args.onnx.resolve())
    sess = create_ort_session(onnx_path)
    print(f"Model: {onnx_path}")
    print(f"Providers: {sess.get_providers()}")

    src = args.image_or_video
    if src.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"Failed to read video: {src}")
        img = frame
    else:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"Failed to read image: {src}")

    poses = infer_rtmo_s(img, sess, args.score_thr)
    print(f"detected {len(poses)} person(s)")
    for i, p in enumerate(poses):
        print(f"  person {i}: score={p['score']:.3f} box={p['box_xyxy']}")

    vis = draw_poses(img, poses)
    cv2.imwrite(args.out, vis)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

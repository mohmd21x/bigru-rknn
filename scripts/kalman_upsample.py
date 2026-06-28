#!/usr/bin/env python3
"""Upsample sparse keypoint CSVs to a regular target FPS via Kalman smoothing.

Each COCO-17 keypoint is tracked independently with a 4D constant-velocity Kalman
filter (state: x, y, vx, vy).

``causal`` mode (default) runs a forward-only filter with past-only absence
detection and exponential confidence decay — safe for realtime production.

``rts`` mode applies a Rauch-Tung-Striebel backward smoother and bilateral
absence detection — offline visualization only (uses future measurements).
"""

from __future__ import annotations

import argparse
import csv
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

NUM_KEYPOINTS = 17
CSV_HEADER = ["video_name", "frame_index", "timestamp", "person_id"]
for _i in range(NUM_KEYPOINTS):
    CSV_HEADER.extend([f"kpt{_i}_x", f"kpt{_i}_y", f"kpt{_i}_conf"])

MIN_CONF = 1e-3


class KalmanFilter4D:
    """Constant-velocity Kalman filter for a single (x, y) keypoint."""

    def __init__(self, process_noise: float, meas_noise_base: float) -> None:
        self.process_noise = float(process_noise)
        self.meas_noise_base = float(meas_noise_base)
        self.x = np.zeros(4, dtype=np.float64)
        self.p = np.eye(4, dtype=np.float64) * 1e3
        self.initialized = False

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _process_covariance(self, dt: float) -> np.ndarray:
        q = self.process_noise**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _observation_matrix() -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    def initialize(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.p = np.eye(4, dtype=np.float64) * 1e2
        self.initialized = True

    def set_state(self, x: np.ndarray, p: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.p = np.asarray(p, dtype=np.float64).copy()
        self.initialized = True

    def reset(self) -> None:
        """Fully discard state. Next update() will call initialize()."""
        self.initialized = False
        self.x = np.zeros(4, dtype=np.float64)
        self.p = np.eye(4, dtype=np.float64) * 1e3

    def predict(self, dt: float) -> None:
        if dt <= 0.0 or not self.initialized:
            return
        f_mat = self._transition(dt)
        self.x = f_mat @ self.x
        self.p = f_mat @ self.p @ f_mat.T + self._process_covariance(dt)

    def predict_store(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict forward by ``dt``; return (x_pred, P_pred, F) and update state."""
        if dt <= 0.0 or not self.initialized:
            identity = np.eye(4, dtype=np.float64)
            return self.x.copy(), self.p.copy(), identity

        f_mat = self._transition(dt)
        x_pred = f_mat @ self.x
        p_pred = f_mat @ self.p @ f_mat.T + self._process_covariance(dt)
        self.x = x_pred
        self.p = p_pred
        return x_pred.copy(), p_pred.copy(), f_mat

    def update(self, measurement: np.ndarray, confidence: float) -> None:
        if not self.initialized:
            self.initialize(float(measurement[0]), float(measurement[1]))
            return

        h_mat = self._observation_matrix()
        sigma2 = self.meas_noise_base**2
        conf = max(float(confidence), MIN_CONF)
        r_mat = np.diag([sigma2 / conf, sigma2 / conf])

        innovation = measurement - (h_mat @ self.x)
        innovation_cov = h_mat @ self.p @ h_mat.T + r_mat
        kalman_gain = self.p @ h_mat.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.p = (identity - kalman_gain @ h_mat) @ self.p

    def update_store(
        self,
        measurement: np.ndarray,
        confidence: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a measurement update; return (x_filt, P_filt) and update state."""
        if not self.initialized:
            self.initialize(float(measurement[0]), float(measurement[1]))
            return self.x.copy(), self.p.copy()

        h_mat = self._observation_matrix()
        sigma2 = self.meas_noise_base**2
        conf = max(float(confidence), MIN_CONF)
        r_mat = np.diag([sigma2 / conf, sigma2 / conf])

        innovation = measurement - (h_mat @ self.x)
        innovation_cov = h_mat @ self.p @ h_mat.T + r_mat
        kalman_gain = self.p @ h_mat.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.p = (identity - kalman_gain @ h_mat) @ self.p
        return self.x.copy(), self.p.copy()

    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


def parse_keypoints(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy, conf) arrays with shape (17, 2) and (17,)."""
    xy = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
    for i in range(NUM_KEYPOINTS):
        xy[i, 0] = float(row[f"kpt{i}_x"])
        xy[i, 1] = float(row[f"kpt{i}_y"])
        conf[i] = float(row[f"kpt{i}_conf"])
    return xy, conf


def is_valid_measurement(x: float, y: float, conf: float) -> bool:
    return conf > MIN_CONF and (abs(x) > 1e-6 or abs(y) > 1e-6)


def _time_key(t: float) -> float:
    return round(float(t), 9)


def _build_merged_times(
    rows: list[dict[str, str]],
    output_times: np.ndarray,
) -> tuple[list[float], dict[float, tuple[np.ndarray, np.ndarray]]]:
    meas_at_time: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        t = _time_key(float(row["timestamp"]))
        meas_at_time[t] = parse_keypoints(row)

    merged = sorted(set(output_times.tolist()) | set(meas_at_time.keys()))
    return merged, meas_at_time


def _rts_smooth(
    x_filt: list[np.ndarray],
    p_filt: list[np.ndarray],
    x_pred: list[np.ndarray | None],
    p_pred: list[np.ndarray | None],
    f_mats: list[np.ndarray | None],
) -> list[np.ndarray]:
    """Rauch-Tung-Striebel backward smoother."""
    n = len(x_filt)
    x_smooth: list[np.ndarray] = [arr.copy() for arr in x_filt]

    for idx in range(n - 2, -1, -1):
        f_next = f_mats[idx + 1]
        x_p_next = x_pred[idx + 1]
        p_p_next = p_pred[idx + 1]
        if f_next is None or x_p_next is None or p_p_next is None:
            continue

        gain = p_filt[idx] @ f_next.T @ np.linalg.inv(p_p_next)
        x_smooth[idx] = x_filt[idx] + gain @ (x_smooth[idx + 1] - x_p_next)

    return x_smooth


def _forward_pass_keypoint(
    merged_times: list[float],
    meas_at_time: dict[float, tuple[np.ndarray, np.ndarray]],
    k: int,
    process_noise: float,
    meas_noise_base: float,
) -> list[np.ndarray]:
    filt = KalmanFilter4D(process_noise, meas_noise_base)
    x_filt: list[np.ndarray] = []
    p_filt: list[np.ndarray] = []
    x_pred: list[np.ndarray | None] = [None]
    p_pred: list[np.ndarray | None] = [None]
    f_mats: list[np.ndarray | None] = [None]

    for step_idx, t in enumerate(merged_times):
        xy: np.ndarray | None = None
        conf_k = 0.0
        if t in meas_at_time:
            xy, conf = meas_at_time[t]
            if is_valid_measurement(xy[k, 0], xy[k, 1], conf[k]):
                xy_k = xy[k]
                conf_k = float(conf[k])

        if step_idx == 0:
            if xy is not None and conf_k > MIN_CONF:
                filt.initialize(float(xy[k, 0]), float(xy[k, 1]))
                x_filt.append(filt.x.copy())
                p_filt.append(filt.p.copy())
            else:
                x_filt.append(np.zeros(4, dtype=np.float64))
                p_filt.append(np.eye(4, dtype=np.float64) * 1e3)
            continue

        dt = merged_times[step_idx] - merged_times[step_idx - 1]
        x_p, p_p, f_mat = filt.predict_store(dt)
        x_pred.append(x_p)
        p_pred.append(p_p)
        f_mats.append(f_mat)

        if xy is not None and conf_k > MIN_CONF:
            x_f, p_f = filt.update_store(xy[k], conf_k)
        else:
            x_f, p_f = x_p, p_p
            filt.set_state(x_f, p_f)

        x_filt.append(x_f)
        p_filt.append(p_f)

    if not filt.initialized:
        return [np.zeros(4, dtype=np.float64) for _ in merged_times]

    return _rts_smooth(x_filt, p_filt, x_pred, p_pred, f_mats)


def _collect_valid_measurements(
    rows: list[dict[str, str]],
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return sorted (timestamp, xy, conf) for each input row."""
    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    for row in rows:
        t = float(row["timestamp"])
        xy, conf = parse_keypoints(row)
        samples.append((t, xy, conf))
    samples.sort(key=lambda item: item[0])
    return samples


def _valid_times_for_keypoint(
    samples: list[tuple[float, np.ndarray, np.ndarray]],
    k: int,
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    confs: list[float] = []
    for t, xy, conf in samples:
        if is_valid_measurement(xy[k, 0], xy[k, 1], conf[k]):
            times.append(t)
            confs.append(float(conf[k]))
    return times, confs


def _is_absent(t: float, valid_times: list[float], max_gap_sec: float) -> bool:
    if not valid_times:
        return True

    left_idx = bisect_right(valid_times, t) - 1
    if left_idx < 0:
        gap_left = float("inf")
    else:
        gap_left = t - valid_times[left_idx]

    right_idx = bisect_right(valid_times, t)
    if right_idx >= len(valid_times):
        gap_right = float("inf")
    else:
        gap_right = valid_times[right_idx] - t

    return min(gap_left, gap_right) > max_gap_sec


def _interpolate_conf(
    t: float,
    valid_times: list[float],
    valid_confs: list[float],
) -> float:
    if not valid_times:
        return 0.0

    left_idx = bisect_right(valid_times, t) - 1
    right_idx = bisect_right(valid_times, t)

    if left_idx >= 0 and abs(valid_times[left_idx] - t) < 1e-9:
        return valid_confs[left_idx]

    if left_idx < 0:
        return valid_confs[right_idx] if right_idx < len(valid_times) else 0.0
    if right_idx >= len(valid_times):
        return valid_confs[left_idx]

    t0, t1 = valid_times[left_idx], valid_times[right_idx]
    if t1 <= t0 + 1e-12:
        return valid_confs[left_idx]

    alpha = (t - t0) / (t1 - t0)
    return valid_confs[left_idx] + alpha * (valid_confs[right_idx] - valid_confs[left_idx])


def _is_absent_causal(t: float, last_valid_t: float | None, max_gap_sec: float) -> bool:
    if last_valid_t is None:
        return True
    return (t - last_valid_t) > max_gap_sec


def _conf_decay(last_conf: float, gap: float, conf_decay_sec: float) -> float:
    if conf_decay_sec <= 0.0:
        return last_conf
    return last_conf * float(np.exp(-gap / conf_decay_sec))


def causal_upsample_measurements(
    measurements: list[tuple[float, np.ndarray | None]],
    t_start: float,
    t_end: float,
    *,
    target_fps: float = 30.0,
    process_noise: float = 20.0,
    meas_noise_base: float = 10.0,
    max_gap_sec: float = 0.5,
    conf_decay_sec: float = 0.3,
) -> list[tuple[float, np.ndarray]]:
    """Forward-only Kalman upsample for realtime use (no future leakage).

    ``measurements`` is a list of ``(timestamp, keypoints)`` sorted or unsorted.
    ``keypoints`` may be ``None`` when YOLO found no person at that sample time.
    Returns ``(timestamp, 17x3)`` tuples on a regular ``target_fps`` grid.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    if max_gap_sec <= 0:
        raise ValueError("max_gap_sec must be > 0")
    if conf_decay_sec <= 0:
        raise ValueError("conf_decay_sec must be > 0")

    dt_out = 1.0 / target_fps
    output_times = np.arange(t_start, t_end + dt_out * 0.5, dt_out)
    sorted_meas = sorted(measurements, key=lambda item: item[0])

    filters = [KalmanFilter4D(process_noise, meas_noise_base) for _ in range(NUM_KEYPOINTS)]
    filter_time = float(t_start)
    last_valid_t: list[float | None] = [None] * NUM_KEYPOINTS
    last_valid_conf = [0.0] * NUM_KEYPOINTS

    meas_idx = 0
    results: list[tuple[float, np.ndarray]] = []

    for t_out in output_times:
        t_out_f = float(t_out)

        while meas_idx < len(sorted_meas) and sorted_meas[meas_idx][0] <= t_out_f + 1e-9:
            t_meas, kpts = sorted_meas[meas_idx]
            if t_meas > filter_time + 1e-12:
                dt_meas = t_meas - filter_time
                for filt in filters:
                    filt.predict(dt_meas)
                filter_time = t_meas

            if kpts is not None:
                kpts_arr = np.asarray(kpts, dtype=np.float64).reshape(NUM_KEYPOINTS, 3)
                for k in range(NUM_KEYPOINTS):
                    x, y, conf = kpts_arr[k, 0], kpts_arr[k, 1], kpts_arr[k, 2]
                    if is_valid_measurement(x, y, conf):
                        gap_since_valid = (
                            t_meas - last_valid_t[k]
                            if last_valid_t[k] is not None
                            else float("inf")
                        )
                        if (
                            not filters[k].initialized
                            or gap_since_valid > max_gap_sec
                        ):
                            filters[k].reset()
                            filters[k].initialize(float(x), float(y))
                        else:
                            filters[k].update(
                                np.array([x, y], dtype=np.float64), conf
                            )
                        last_valid_t[k] = t_meas
                        last_valid_conf[k] = float(conf)

            meas_idx += 1

        if t_out_f > filter_time + 1e-12:
            dt_out_step = t_out_f - filter_time
            for filt in filters:
                filt.predict(dt_out_step)
            filter_time = t_out_f

        out_kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        for k in range(NUM_KEYPOINTS):
            if _is_absent_causal(t_out_f, last_valid_t[k], max_gap_sec):
                continue
            gap = t_out_f - float(last_valid_t[k])
            x_pos, y_pos = filters[k].position()
            out_kpts[k, 0] = x_pos
            out_kpts[k, 1] = y_pos
            out_kpts[k, 2] = _conf_decay(last_valid_conf[k], gap, conf_decay_sec)

        results.append((t_out_f, out_kpts))

    return results


def upsample_track_causal(
    rows: list[dict[str, str]],
    *,
    target_fps: float,
    process_noise: float,
    meas_noise_base: float,
    max_gap_sec: float,
    conf_decay_sec: float,
) -> list[list[float | int | str]]:
    """Upsample one track with forward-only Kalman filtering (production-safe)."""
    rows = sorted(rows, key=lambda row: float(row["timestamp"]))
    if not rows:
        return []

    video_name = rows[0]["video_name"]
    person_id = int(float(rows[0]["person_id"]))
    t_start = float(rows[0]["timestamp"])
    t_end = float(rows[-1]["timestamp"])

    measurements: list[tuple[float, np.ndarray | None]] = []
    for row in rows:
        xy, conf = parse_keypoints(row)
        kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        kpts[:, :2] = xy
        kpts[:, 2] = conf
        measurements.append((float(row["timestamp"]), kpts))

    upsampled = causal_upsample_measurements(
        measurements,
        t_start,
        t_end,
        target_fps=target_fps,
        process_noise=process_noise,
        meas_noise_base=meas_noise_base,
        max_gap_sec=max_gap_sec,
        conf_decay_sec=conf_decay_sec,
    )

    output_rows: list[list[float | int | str]] = []
    for frame_index, (t_out, kpts) in enumerate(upsampled):
        row: list[float | int | str] = [video_name, frame_index, t_out, person_id]
        for k in range(NUM_KEYPOINTS):
            row.extend([float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])])
        output_rows.append(row)

    return output_rows


def upsample_track(
    rows: list[dict[str, str]],
    *,
    target_fps: float,
    process_noise: float,
    meas_noise_base: float,
    max_gap_sec: float,
) -> list[list[float | int | str]]:
    """Upsample one (video_name, person_id) track to regular target_fps."""
    rows = sorted(rows, key=lambda row: float(row["timestamp"]))
    if not rows:
        return []

    video_name = rows[0]["video_name"]
    person_id = int(float(rows[0]["person_id"]))
    t_start = float(rows[0]["timestamp"])
    t_end = float(rows[-1]["timestamp"])
    dt_out = 1.0 / target_fps

    output_times = np.arange(t_start, t_end + dt_out * 0.5, dt_out)
    merged_times, meas_at_time = _build_merged_times(rows, output_times)
    merged_index = {_time_key(t): idx for idx, t in enumerate(merged_times)}

    samples = _collect_valid_measurements(rows)

    smooth_states: list[list[np.ndarray]] = []
    valid_times_per_kpt: list[list[float]] = []
    valid_confs_per_kpt: list[list[float]] = []

    for k in range(NUM_KEYPOINTS):
        smooth_states.append(
            _forward_pass_keypoint(
                merged_times,
                meas_at_time,
                k,
                process_noise,
                meas_noise_base,
            )
        )
        times_k, confs_k = _valid_times_for_keypoint(samples, k)
        valid_times_per_kpt.append(times_k)
        valid_confs_per_kpt.append(confs_k)

    output_rows: list[list[float | int | str]] = []
    for frame_index, t_out in enumerate(output_times):
        t_key = _time_key(float(t_out))
        step_idx = merged_index[t_key]

        row: list[float | int | str] = [video_name, frame_index, float(t_out), person_id]
        for k in range(NUM_KEYPOINTS):
            if _is_absent(float(t_out), valid_times_per_kpt[k], max_gap_sec):
                row.extend([0.0, 0.0, 0.0])
                continue

            x_smooth, y_smooth = (
                float(smooth_states[k][step_idx][0]),
                float(smooth_states[k][step_idx][1]),
            )
            conf_out = _interpolate_conf(
                float(t_out),
                valid_times_per_kpt[k],
                valid_confs_per_kpt[k],
            )
            row.extend([x_smooth, y_smooth, conf_out])

        output_rows.append(row)

    return output_rows


def read_keypoint_csv(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv}")
        missing = [col for col in CSV_HEADER if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {input_csv} missing columns: {missing}")
        return list(reader)


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["video_name"], int(float(row["person_id"])))
        groups[key].append(row)
    return groups


def upsample_keypoint_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    target_fps: float = 30.0,
    process_noise: float | None = None,
    meas_noise_base: float | None = None,
    max_gap_sec: float | None = None,
    conf_decay_sec: float = 0.3,
    mode: str = "causal",
) -> Path:
    """Upsample a sparse keypoint CSV to a regular target FPS."""
    if mode not in {"causal", "rts"}:
        raise ValueError("mode must be 'causal' or 'rts'")

    if mode == "causal":
        process_noise = 20.0 if process_noise is None else process_noise
        meas_noise_base = 10.0 if meas_noise_base is None else meas_noise_base
        max_gap_sec = 0.5 if max_gap_sec is None else max_gap_sec
    else:
        process_noise = 5.0 if process_noise is None else process_noise
        meas_noise_base = 20.0 if meas_noise_base is None else meas_noise_base
        max_gap_sec = 1.0 if max_gap_sec is None else max_gap_sec

    if target_fps <= 0:
        raise ValueError("--target-fps must be > 0")
    if process_noise < 0 or meas_noise_base <= 0:
        raise ValueError("--process-noise must be >= 0 and --meas-noise-base must be > 0")
    if max_gap_sec <= 0:
        raise ValueError("--max-gap-sec must be > 0")
    if conf_decay_sec <= 0:
        raise ValueError("--conf-decay-sec must be > 0")

    input_path = input_csv.resolve()
    rows = read_keypoint_csv(input_path)
    if not rows:
        raise ValueError(f"input CSV is empty: {input_path}")

    groups = group_rows(rows)
    output_rows: list[list[float | int | str]] = []
    for group_rows_list in groups.values():
        if mode == "causal":
            output_rows.extend(
                upsample_track_causal(
                    group_rows_list,
                    target_fps=target_fps,
                    process_noise=process_noise,
                    meas_noise_base=meas_noise_base,
                    max_gap_sec=max_gap_sec,
                    conf_decay_sec=conf_decay_sec,
                )
            )
        else:
            output_rows.extend(
                upsample_track(
                    group_rows_list,
                    target_fps=target_fps,
                    process_noise=process_noise,
                    meas_noise_base=meas_noise_base,
                    max_gap_sec=max_gap_sec,
                )
            )

    output_rows.sort(key=lambda row: (str(row[0]), int(row[3]), float(row[2])))

    output_path = output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(output_rows)

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upsample sparse keypoint CSV to regular FPS via Kalman filtering.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input sparse keypoint CSV (e.g. ~7 FPS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output upsampled keypoint CSV.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Output sampling rate (default: 30).",
    )
    parser.add_argument(
        "--mode",
        choices=("causal", "rts"),
        default="causal",
        help="causal: forward-only (production); rts: offline RTS smoother (default: causal).",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=None,
        help="Process noise sigma (default: 20 causal, 5 rts).",
    )
    parser.add_argument(
        "--meas-noise-base",
        type=float,
        default=None,
        help="Base measurement noise sigma; scaled by 1/confidence (default: 10 causal, 20 rts).",
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=None,
        help="Absence gap in seconds (default: 0.5 causal, 1.0 rts).",
    )
    parser.add_argument(
        "--conf-decay-sec",
        type=float,
        default=0.3,
        help="Confidence decay time constant for causal mode (default: 0.3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.target_fps <= 0:
        print("Error: --target-fps must be > 0", file=sys.stderr)
        return 1
    if args.process_noise is not None and args.process_noise < 0:
        print("Error: --process-noise must be >= 0", file=sys.stderr)
        return 1
    if args.meas_noise_base is not None and args.meas_noise_base <= 0:
        print("Error: --meas-noise-base must be > 0", file=sys.stderr)
        return 1
    if args.max_gap_sec is not None and args.max_gap_sec <= 0:
        print("Error: --max-gap-sec must be > 0", file=sys.stderr)
        return 1
    if args.conf_decay_sec <= 0:
        print("Error: --conf-decay-sec must be > 0", file=sys.stderr)
        return 1

    input_csv = args.input.resolve()
    if not input_csv.is_file():
        print(f"Error: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    try:
        rows = read_keypoint_csv(input_csv)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"Error: input CSV is empty: {input_csv}", file=sys.stderr)
        return 1

    try:
        output_csv = upsample_keypoint_csv(
            input_csv,
            args.output,
            target_fps=args.target_fps,
            process_noise=args.process_noise,
            meas_noise_base=args.meas_noise_base,
            max_gap_sec=args.max_gap_sec,
            conf_decay_sec=args.conf_decay_sec,
            mode=args.mode,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    in_count = len(rows)
    out_count = sum(1 for _ in output_csv.open()) - 1
    print(f"Input:  {input_csv} ({in_count} rows)")
    print(f"Output: {output_csv} ({out_count} rows @ {args.target_fps:.1f} fps)")
    print(f"Tracks: {len(group_rows(rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

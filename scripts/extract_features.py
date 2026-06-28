#!/usr/bin/env python3
"""Batch pose feature extraction via the C++ extractor binary.

Reads keypoint CSVs from ``dataset/outputs/`` (or filenames listed in split
manifests) and writes feature CSVs to ``dataset/features/{stem}_features.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXTRACTOR = REPO_ROOT / "pose_features/build/extract_pose_features_from_csv"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "dataset/outputs"
DEFAULT_FEATURES_DIR = REPO_ROOT / "dataset/features"
DEFAULT_SPLITS_DIR = REPO_ROOT / "dataset/splits"
DEFAULT_ERROR_LOG = DEFAULT_FEATURES_DIR / "extract_errors.log"

# Common ONNX Runtime install roots (see pose_features/CMakeLists.txt).
ONNXRUNTIME_LIB_CANDIDATES: tuple[Path, ...] = (
    Path.home() / ".local/onnxruntime-linux-x64-gpu-1.19.2/lib",
    Path.home() / ".local/onnxruntime-linux-x64-1.19.2/lib",
    Path.home() / ".local/onnxruntime-linux-aarch64-1.19.2/lib",
)


def feature_output_path(input_csv: Path, features_dir: Path) -> Path:
    """Map ``foo_keypoints.csv`` -> ``foo_keypoints_features.csv``."""
    return features_dir / f"{input_csv.stem}_features.csv"


def discover_onnxruntime_lib_dir(explicit: Path | None) -> Path | None:
    """Return an ONNX Runtime ``lib`` directory when the extractor needs it.

    CSV-only builds of ``extract_pose_features_from_csv`` do not link ONNX Runtime.
    This helper remains for older binaries that still require ``libonnxruntime.so.1``.
    """
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    for candidate in ONNXRUNTIME_LIB_CANDIDATES:
        if (candidate / "libonnxruntime.so.1").exists() or any(
            candidate.glob("libonnxruntime.so*")
        ):
            return candidate
    return None


def extractor_needs_onnxruntime(extractor: Path) -> bool:
    """Return True when the extractor binary is dynamically linked to ONNX Runtime."""
    try:
        result = subprocess.run(
            ["ldd", str(extractor)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return True
    if result.returncode != 0:
        return True
    return "libonnxruntime" in (result.stdout or "")


def build_subprocess_env(lib_dir: Path | None) -> dict[str, str]:
    """Copy the current environment and prepend ONNX Runtime to ``LD_LIBRARY_PATH``."""
    env = os.environ.copy()
    if lib_dir is None:
        return env
    existing = env.get("LD_LIBRARY_PATH", "")
    prefix = str(lib_dir)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


def read_manifest_filenames(splits_dir: Path) -> list[str]:
    """Collect unique ``filename`` values from train/test split CSVs."""
    filenames: set[str] = set()
    for manifest_name in ("train.csv", "test.csv"):
        manifest_path = splits_dir / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "filename" not in reader.fieldnames:
                raise ValueError(
                    f"Manifest {manifest_path} must contain a 'filename' column"
                )
            for row in reader:
                name = (row.get("filename") or "").strip()
                if name:
                    filenames.add(name)
    return sorted(filenames)


def collect_input_files(
    outputs_dir: Path,
    *,
    from_manifests: bool,
    splits_dir: Path,
) -> list[Path]:
    """Resolve input keypoint CSV paths to process."""
    if from_manifests:
        names = read_manifest_filenames(splits_dir)
        paths = [outputs_dir / name for name in names]
    else:
        paths = sorted(outputs_dir.glob("*.csv"))
    return paths


def append_error_log(log_path: Path, input_csv: Path, message: str) -> None:
    """Append one failure line to the extract error log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\t{input_csv}\t{message}\n")


def run_extractor(
    extractor: Path,
    input_csv: Path,
    output_csv: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Invoke the C++ extractor for a single input/output pair."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [str(extractor), str(input_csv), str(output_csv)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
    except OSError as exc:
        return False, str(exc)

    if result.returncode == 0:
        if not output_csv.is_file() or output_csv.stat().st_size == 0:
            return False, "extractor exited 0 but output file is missing or empty"
        return True, ""

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"exit code {result.returncode}"
    return False, detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pose features from keypoint CSVs using the C++ binary.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory containing raw keypoint CSV files (default: dataset/outputs).",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory for feature CSV output (default: dataset/features).",
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=DEFAULT_EXTRACTOR,
        help="Path to extract_pose_features_from_csv binary.",
    )
    parser.add_argument(
        "--extractor-lib-dir",
        type=Path,
        default=None,
        help="ONNX Runtime lib directory for LD_LIBRARY_PATH (auto-detected if omitted).",
    )
    parser.add_argument(
        "--from-manifests",
        action="store_true",
        help="Process only filenames listed in dataset/splits/train.csv and test.csv.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory with train.csv and test.csv (used with --from-manifests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inputs whose feature CSV already exists and is non-empty.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N files (useful for smoke tests).",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=DEFAULT_ERROR_LOG,
        help="Append extraction failures to this log file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    extractor = args.extractor.resolve()
    if not extractor.is_file():
        print(f"Error: extractor binary not found: {extractor}", file=sys.stderr)
        print(
            "Rebuild with: cd pose_features && mkdir -p build && cd build && "
            "cmake .. && cmake --build .",
            file=sys.stderr,
        )
        return 1

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"Error: outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    features_dir = args.features_dir.resolve()
    features_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_files = collect_input_files(
            outputs_dir,
            from_manifests=args.from_manifests,
            splits_dir=args.splits_dir.resolve(),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit < 0:
            print("Error: --limit must be >= 0", file=sys.stderr)
            return 1
        input_files = input_files[: args.limit]

    lib_dir = discover_onnxruntime_lib_dir(
        args.extractor_lib_dir.resolve() if args.extractor_lib_dir else None
    )
    if extractor_needs_onnxruntime(extractor) and lib_dir is None:
        print(
            "Error: extractor requires ONNX Runtime 1.19.x but no library was found.",
            file=sys.stderr,
        )
        print(
            "Install to ~/.local/onnxruntime-linux-x64-gpu-1.19.2 or pass "
            "--extractor-lib-dir, or rebuild without ONNX:",
            file=sys.stderr,
        )
        print(
            "  cd pose_features && rm -rf build && mkdir build && cd build && "
            "cmake .. && cmake --build .",
            file=sys.stderr,
        )
        return 1
    env = build_subprocess_env(lib_dir)

    stats = {"processed": 0, "skipped": 0, "failed": 0, "missing_input": 0}

    for input_csv in tqdm(input_files, desc="Extracting features", unit="file"):
        if not input_csv.is_file():
            stats["missing_input"] += 1
            message = "input file not found"
            append_error_log(args.error_log.resolve(), input_csv, message)
            tqdm.write(f"Missing input: {input_csv}")
            continue

        output_csv = feature_output_path(input_csv, features_dir)
        if args.skip_existing and output_csv.is_file() and output_csv.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        ok, error = run_extractor(extractor, input_csv, output_csv, env)
        if ok:
            stats["processed"] += 1
        else:
            stats["failed"] += 1
            append_error_log(args.error_log.resolve(), input_csv, error)
            tqdm.write(f"Failed: {input_csv.name} ({error})")

    print(
        "Done. "
        f"processed={stats['processed']} "
        f"skipped={stats['skipped']} "
        f"failed={stats['failed']} "
        f"missing_input={stats['missing_input']}"
    )
    if stats["failed"] or stats["missing_input"]:
        print(f"See error log: {args.error_log.resolve()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resample keypoint CSVs into phase-shifted ~7.5 FPS variants.

Reads train/test manifests from ``dataset/splits`` and writes:
1) Phase-subsampled keypoint CSVs to ``dataset/outputs_7fps``.
2) Expanded train/test manifests to ``dataset/splits_7fps``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "dataset/outputs"
DEFAULT_SPLITS_DIR = REPO_ROOT / "dataset/splits"
DEFAULT_OUT_OUTPUTS_DIR = REPO_ROOT / "dataset/outputs_7fps"
DEFAULT_OUT_SPLITS_DIR = REPO_ROOT / "dataset/splits_7fps"


def with_phase_suffix(filename: str, phase: int) -> str:
    """Insert ``_ph{phase}`` before ``_keypoints.csv`` when possible."""
    suffix = "_keypoints.csv"
    if filename.endswith(suffix):
        return f"{filename[: -len(suffix)]}_ph{phase}{suffix}"
    stem = Path(filename).stem
    ext = Path(filename).suffix or ".csv"
    return f"{stem}_ph{phase}{ext}"


def resolve_input_csv(outputs_dir: Path, row: dict[str, str]) -> Path:
    """Resolve input CSV path, preferring the local outputs directory."""
    filename = (row.get("filename") or "").strip()
    if not filename:
        raise ValueError("Manifest row missing 'filename'.")

    from_outputs = outputs_dir / filename
    if from_outputs.is_file():
        return from_outputs

    path_value = (row.get("path") or "").strip()
    if path_value:
        raw_path = Path(path_value)
        if raw_path.is_file():
            return raw_path

    raise FileNotFoundError(f"Input keypoint CSV not found for '{filename}'.")


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        required = {"filename", "path", "label", "split"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Manifest {manifest_path} missing columns: {missing}")
        return list(reader)


def write_subsampled_csv(
    input_csv: Path,
    output_csv: Path,
    phase: int,
    step: int,
    skip_existing: bool = False,
) -> int:
    """Write one phase-subsampled CSV, preserving original columns and values."""
    if skip_existing and output_csv.is_file() and output_csv.stat().st_size > 0:
        return -1
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    header = lines[0]
    data_rows = lines[1:]
    selected_rows = data_rows[phase::step]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        handle.writelines(selected_rows)
    return len(selected_rows)


def process_manifest(
    manifest_name: str,
    *,
    outputs_dir: Path,
    out_outputs_dir: Path,
    splits_dir: Path,
    out_splits_dir: Path,
    step: int,
    phases: int,
    skip_existing: bool = False,
) -> tuple[int, int]:
    """Resample rows from one manifest and write expanded output manifest.

    Returns ``(source_rows, generated_rows)``.
    """
    in_manifest = splits_dir / manifest_name
    out_manifest = out_splits_dir / manifest_name
    rows = read_manifest_rows(in_manifest)

    out_splits_dir.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["filename", "path", "label", "split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        generated_rows = 0
        for row in rows:
            input_csv = resolve_input_csv(outputs_dir, row)
            for phase in range(phases):
                out_filename = with_phase_suffix((row.get("filename") or "").strip(), phase)
                out_csv = out_outputs_dir / out_filename
                write_subsampled_csv(
                    input_csv, out_csv, phase=phase, step=step, skip_existing=skip_existing
                )

                writer.writerow(
                    {
                        "filename": out_filename,
                        "path": str(out_csv.resolve()),
                        "label": (row.get("label") or "").strip(),
                        "split": (row.get("split") or "").strip(),
                    }
                )
                generated_rows += 1

    return len(rows), generated_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create phase-shifted subsampled keypoint CSVs and manifests.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Input keypoint CSV directory (default: dataset/outputs).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Input manifest directory containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--out-outputs-dir",
        type=Path,
        default=DEFAULT_OUT_OUTPUTS_DIR,
        help="Output directory for phase-resampled keypoint CSVs.",
    )
    parser.add_argument(
        "--out-splits-dir",
        type=Path,
        default=DEFAULT_OUT_SPLITS_DIR,
        help="Output manifest directory (train.csv and test.csv).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Subsampling step size (default: 4).",
    )
    parser.add_argument(
        "--phases",
        type=int,
        default=4,
        help="Number of phase offsets to generate (default: 4).",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="*",
        default=["train.csv", "val.csv", "test.csv"],
        help="Manifest filenames to expand (missing ones are skipped).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing keypoint CSVs that already exist (regenerate manifests only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step <= 0 or args.phases <= 0:
        print("Error: --step and --phases must be > 0.", file=sys.stderr)
        return 1

    outputs_dir = args.outputs_dir.resolve()
    splits_dir = args.splits_dir.resolve()
    out_outputs_dir = args.out_outputs_dir.resolve()
    out_splits_dir = args.out_splits_dir.resolve()

    if not outputs_dir.is_dir():
        print(f"Error: outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1
    if not splits_dir.is_dir():
        print(f"Error: splits directory not found: {splits_dir}", file=sys.stderr)
        return 1

    total_in = 0
    total_out = 0
    for manifest_name in args.manifests:
        if not (splits_dir / manifest_name).is_file():
            print(f"Skipping missing manifest: {manifest_name}")
            continue
        try:
            in_rows, out_rows = process_manifest(
                manifest_name,
                outputs_dir=outputs_dir,
                out_outputs_dir=out_outputs_dir,
                splits_dir=splits_dir,
                out_splits_dir=out_splits_dir,
                step=args.step,
                phases=args.phases,
                skip_existing=args.skip_existing,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        total_in += in_rows
        total_out += out_rows
        print(f"{manifest_name}: {in_rows} -> {out_rows} rows")

    print(f"Done. Generated {total_out} manifest rows from {total_in} source rows.")
    print(f"Outputs: {out_outputs_dir}")
    print(f"Splits: {out_splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

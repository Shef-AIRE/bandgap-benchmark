#!/usr/bin/env python3
"""
Download CIF files from Materials Project by mp-id.

Examples:
  python3 download_cif_by_mpid.py --mpids mp-149 mp-13 --out-dir cif_file
  python3 download_cif_by_mpid.py --input data/stratified_split/stratified_test_formulas.csv --column material_id
  python3 download_cif_by_mpid.py --input mpids.txt --api-key YOUR_KEY --retries 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import getpass
from pathlib import Path
from typing import Iterable

MPID_PATTERN = re.compile(r"^mp-\d+$")


def _normalize_mpid(value: str) -> str | None:
    s = str(value).strip()
    return s if MPID_PATTERN.match(s) else None


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _read_from_txt(path: Path) -> list[str]:
    mpids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # accept comma or whitespace separated
        parts = re.split(r"[\s,]+", line)
        for part in parts:
            if not part:
                continue
            mpid = _normalize_mpid(part)
            if mpid:
                mpids.append(mpid)
    return mpids


def _read_from_csv(path: Path, column: str | None) -> list[str]:
    mpids: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mpids
        candidate_cols = [column] if column else ["material_id", "mpid", "mpids", "id"]
        selected_col = None
        for c in candidate_cols:
            if c and c in reader.fieldnames:
                selected_col = c
                break
        if not selected_col:
            raise ValueError(
                f"Cannot find mp-id column in {path}. "
                f"Pass --column explicitly. Available columns: {reader.fieldnames}"
            )
        for row in reader:
            mpid = _normalize_mpid(row.get(selected_col, ""))
            if mpid:
                mpids.append(mpid)
    return mpids


def _read_from_json(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    mpids: list[str] = []
    if isinstance(obj, dict):
        # support dict keyed by mp-id
        for k in obj.keys():
            mpid = _normalize_mpid(k)
            if mpid:
                mpids.append(mpid)
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, str):
                mpid = _normalize_mpid(x)
            elif isinstance(x, dict):
                mpid = _normalize_mpid(
                    x.get("material_id") or x.get("mpid") or x.get("id") or ""
                )
            else:
                mpid = None
            if mpid:
                mpids.append(mpid)
    return mpids


def load_mpids(input_path: Path | None, direct_mpids: list[str], column: str | None) -> list[str]:
    mpids: list[str] = []

    for m in direct_mpids:
        mpid = _normalize_mpid(m)
        if mpid:
            mpids.append(mpid)

    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        suffix = input_path.suffix.lower()
        if suffix in {".txt"}:
            mpids.extend(_read_from_txt(input_path))
        elif suffix in {".csv"}:
            mpids.extend(_read_from_csv(input_path, column))
        elif suffix in {".json"}:
            mpids.extend(_read_from_json(input_path))
        else:
            # fallback: try txt parser
            mpids.extend(_read_from_txt(input_path))

    return _unique_keep_order(mpids)


def download_cifs(
    api_key: str,
    mpids: list[str],
    out_dir: Path,
    retries: int,
    sleep_s: float,
    overwrite: bool,
) -> tuple[list[str], list[str], list[str]]:
    try:
        from mp_api.client import MPRester
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "mp_api is required for CIF download. Install it with: pip install mp-api"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    with MPRester(api_key) as mpr:
        for i, mpid in enumerate(mpids, start=1):
            out_file = out_dir / f"{mpid}.cif"
            if out_file.exists() and not overwrite:
                print(f"[{i}/{len(mpids)}] skip {mpid} (exists)")
                skipped.append(mpid)
                continue

            success = False
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    structure = mpr.get_structure_by_material_id(mpid)
                    if structure is None:
                        raise RuntimeError("No structure returned")
                    structure.to(fmt="cif", filename=str(out_file))
                    print(f"[{i}/{len(mpids)}] ok   {mpid}")
                    ok.append(mpid)
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    if attempt < retries:
                        print(
                            f"[{i}/{len(mpids)}] retry {mpid} "
                            f"(attempt {attempt}/{retries}, err: {exc})"
                        )
                        time.sleep(sleep_s)
                    else:
                        print(f"[{i}/{len(mpids)}] fail  {mpid} (err: {exc})")
            if not success:
                failed.append(mpid)

    return ok, skipped, failed


def auto_download_missing_cifs(
    mpids: Iterable[str],
    out_dir: Path | str,
    api_key: str = "",
    retries: int = 2,
    sleep_s: float = 0.8,
    auto_download: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Ensure all CIFs exist; auto-download missing ones."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized = []
    for m in mpids:
        mpid = _normalize_mpid(str(m))
        if mpid:
            normalized.append(mpid)
    normalized = _unique_keep_order(normalized)

    missing = [mpid for mpid in normalized if not (out_dir / f"{mpid}.cif").exists()]
    if not missing:
        return [], [], []

    print(f"Found {len(missing)} missing CIF file(s).")
    if not auto_download:
        preview = ", ".join(missing[:8])
        raise FileNotFoundError(
            f"Missing CIF files under {out_dir}. Example mp-ids: {preview}. "
            "Enable auto_download and provide api_key (or MP_API_KEY)."
        )

    api_key = (api_key or "").strip()
    if not api_key:
        print("MP API key is required to auto-download missing CIFs.")
        api_key = prompt_api_key()
    if not api_key:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "MP API key is still missing. "
            "Pass api_key or set MP_API_KEY. "
            f"Example missing mp-ids: {preview}"
        )

    ok, skipped, failed = download_cifs(
        api_key=api_key,
        mpids=missing,
        out_dir=out_dir,
        retries=max(1, int(retries)),
        sleep_s=max(0.0, float(sleep_s)),
        overwrite=False,
    )
    print(f"CIF download summary: ok={len(ok)}, skipped={len(skipped)}, failed={len(failed)}")
    if failed:
        preview = ", ".join(failed[:8])
        raise RuntimeError(f"Failed to download {len(failed)} CIF file(s). Example mp-ids: {preview}")
    return ok, skipped, failed


def auto_download_missing_cifs_from_frame(
    frame,
    out_dir: Path | str,
    mpid_col: str = "mpids",
    api_key: str = "",
    retries: int = 2,
    sleep_s: float = 0.8,
    auto_download: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract mpids from a dataframe-like object and ensure CIFs are present.
    """
    if frame is None:
        return [], [], []

    if hasattr(frame, "__getitem__"):
        col = frame[mpid_col]
        if hasattr(col, "dropna") and hasattr(col, "astype") and hasattr(col, "tolist"):
            mpids = col.dropna().astype(str).tolist()
        else:
            mpids = [str(x) for x in col if x is not None]
    else:
        mpids = []

    return auto_download_missing_cifs(
        mpids=mpids,
        out_dir=out_dir,
        api_key=api_key,
        retries=retries,
        sleep_s=sleep_s,
        auto_download=auto_download,
    )


# Backward-compatible alias
ensure_cifs_present = auto_download_missing_cifs


def download_cifs_for_data_stage(
    stage: str,
    api_key: str = "",
    out_dir: Path | str = "cif_file",
    retries: int = 2,
    sleep_s: float = 0.8,
    auto_download: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """
    Download CIFs by mpids listed in dataset JSON files.

    stage:
      - "pretrain" -> data/pretrain_data.json
      - "finetune" -> data/fine_tune/train_data.json + data/fine_tune/test_data.json
    """
    stage = str(stage).strip().lower()
    if stage == "pretrain":
        json_paths = [Path("data/pretrain_data.json")]
    elif stage == "finetune":
        json_paths = [
            Path("data/fine_tune/train_data.json"),
            Path("data/fine_tune/test_data.json"),
        ]
    else:
        raise ValueError("stage must be 'pretrain' or 'finetune'")

    all_mpids: list[str] = []
    for path in json_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {path}")
        all_mpids.extend(load_mpids(input_path=path, direct_mpids=[], column=None))

    all_mpids = _unique_keep_order(all_mpids)
    print(f"Stage={stage}: {len(all_mpids)} unique mp-ids from {len(json_paths)} JSON file(s).")
    return auto_download_missing_cifs(
        mpids=all_mpids,
        out_dir=out_dir,
        api_key=api_key or os.getenv("MP_API_KEY", ""),
        retries=retries,
        sleep_s=sleep_s,
        auto_download=auto_download,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIF files by Materials Project mp-id")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["pretrain", "finetune"],
        help="One-click dataset mode: pretrain or finetune.",
    )
    parser.add_argument("--mpids", nargs="*", default=[], help="Direct mp-id list, e.g. mp-149 mp-13")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input file containing mp-ids (.txt/.csv/.json)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default=None,
        help="Column name for mp-id when input is CSV (default auto-detect)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cif_file"),
        help="Output CIF directory (default: cif_file)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("MP_API_KEY", ""),
        help="Materials Project API key (default from MP_API_KEY env var)",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retries per mp-id (default: 2)")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="Sleep seconds between retries (default: 0.8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CIF files",
    )
    parser.add_argument(
        "--failed-out",
        type=Path,
        default=Path("failed_mpids.csv"),
        help="Where to save failed mp-ids (default: failed_mpids.csv)",
    )
    return parser.parse_args()


def prompt_api_key() -> str:
    # Terminal-only prompt.
    if sys.stdin.isatty():
        print("MP API key can be generated at: https://next-gen.materialsproject.org/api")
        print("Note: input may not be echoed; keep typing and press Enter.")
        return getpass.getpass("Enter MP API key: ").strip()
    return ""


def main() -> int:
    args = parse_args()

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = prompt_api_key()
    if not api_key:
        print(
            "ERROR: API key missing. Pass --api-key, set MP_API_KEY, or enter it in terminal. "
            "Get one at https://next-gen.materialsproject.org/api",
            file=sys.stderr,
        )
        return 2

    if args.stage:
        try:
            ok, skipped, failed = download_cifs_for_data_stage(
                stage=args.stage,
                api_key=api_key,
                out_dir=args.out_dir,
                retries=max(1, args.retries),
                sleep_s=max(0.0, args.sleep),
                auto_download=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: stage download failed: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            mpids = load_mpids(args.input, args.mpids, args.column)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to load mp-ids: {exc}", file=sys.stderr)
            return 2

        if not mpids:
            print("ERROR: no valid mp-ids found.", file=sys.stderr)
            return 2

        print(f"Total valid mp-ids: {len(mpids)}")
        ok, skipped, failed = download_cifs(
            api_key=api_key,
            mpids=mpids,
            out_dir=args.out_dir,
            retries=max(1, args.retries),
            sleep_s=max(0.0, args.sleep),
            overwrite=args.overwrite,
        )

    if failed:
        args.failed_out.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["failed_mpid"])
            for m in failed:
                writer.writerow([m])

    print("\nSummary")
    print(f"  downloaded: {len(ok)}")
    print(f"  skipped:    {len(skipped)}")
    print(f"  failed:     {len(failed)}")
    if failed:
        print(f"  failed list: {args.failed_out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

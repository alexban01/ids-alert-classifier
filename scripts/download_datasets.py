#!/usr/bin/env python3
"""
download_datasets.py — fetch the raw IDS training datasets into datasets/.

The datasets are NOT stored in git (datasets/ is gitignored). This script
reproduces them from their public sources. Sources are also documented in
datasets/SOURCES.txt.

Usage (always use the project venv):
    .venv/bin/python scripts/download_datasets.py --list
    .venv/bin/python scripts/download_datasets.py --all
    .venv/bin/python scripts/download_datasets.py --only unsw,ctu_normal,uwf
    .venv/bin/python scripts/download_datasets.py --all --force   # re-download

Notes:
  * IoT-23 (~8.7 GB) and CTU-13 (~1.9 GB) are large archive downloads.
  * CTU-Malware-Capture is fetched on-demand per scenario by preprocess_zeek.py
    (loaders/loader_ctu_malware.py) — it is not handled here.
  * CICIDS2017 was dropped from training in v7+; it is intentionally not fetched.
  * Datasets already present (non-empty target dir) are skipped unless --force.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

# Resolve project root from this file (scripts/ -> root) so the script works
# regardless of the current working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")


# ── helpers ─────────────────────────────────────────────────────────────────
def _has_files(path: str) -> bool:
    """True if `path` exists and contains at least one regular file."""
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def _progress(blocks: int, block_size: int, total: int) -> None:
    done = blocks * block_size
    if total > 0:
        pct = min(100.0, done * 100.0 / total)
        sys.stdout.write(f"\r    {pct:5.1f}%  ({done/1e6:,.0f} / {total/1e6:,.0f} MB)")
    else:
        sys.stdout.write(f"\r    {done/1e6:,.0f} MB")
    sys.stdout.flush()


def _download(url: str, dest: str) -> None:
    """Stream a single URL to `dest` with a progress indicator."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    GET {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def _extract_tar(archive: str, dest_dir: str) -> None:
    print(f"    extracting {os.path.basename(archive)} -> {dest_dir}/")
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir)  # noqa: S202 — trusted Stratosphere archives


# ── per-dataset downloaders ──────────────────────────────────────────────────
def get_iot23(force: bool) -> None:
    """IoT-23 small labeled conn.logs (~8.7 GB archive)."""
    target = os.path.join(DATA, "iot-23")
    if _has_files(target) and not force:
        print("  [skip] datasets/iot-23/ already populated")
        return
    url = ("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/"
           "iot_23_datasets_small.tar.gz")
    archive = os.path.join(DATA, "iot_23_datasets_small.tar.gz")
    _download(url, archive)
    _extract_tar(archive, target)
    os.remove(archive)
    print("  [ok] IoT-23")


def get_ctu13(force: bool) -> None:
    """CTU-13 binetflow dataset (~1.9 GB archive)."""
    target = os.path.join(DATA, "ctu-13")
    if _has_files(target) and not force:
        print("  [skip] datasets/ctu-13/ already populated")
        return
    url = ("https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/"
           "CTU-13-Dataset.tar.bz2")
    archive = os.path.join(DATA, "CTU-13-Dataset.tar.bz2")
    _download(url, archive)
    _extract_tar(archive, target)        # -> datasets/ctu-13/CTU-13-Dataset/
    os.remove(archive)
    print("  [ok] CTU-13")


def get_unsw(force: bool) -> None:
    """UNSW-NB15 flow parquet from HuggingFace (~175 MB; excludes byte dumps)."""
    target = os.path.join(DATA, "unsw-nb15")
    if _has_files(target) and not force:
        print("  [skip] datasets/unsw-nb15/ already populated")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("  [error] huggingface_hub not installed: .venv/bin/pip install huggingface_hub")
    print("    snapshot_download rdpahalavan/UNSW-NB15 (excluding ~95 GB byte dumps)")
    snapshot_download(
        repo_id="rdpahalavan/UNSW-NB15",
        repo_type="dataset",
        local_dir=target,
        # Packet-Bytes/ and Payload-Bytes/ are each ~95 GB — NEVER fetch them.
        ignore_patterns=["Packet-Bytes/*", "Payload-Bytes/*"],
    )
    print("  [ok] UNSW-NB15")


def get_uwf(force: bool) -> None:
    """UWF-ZeekData24 native Zeek conn.log CSVs (~21 MB, recursive HTTP dir)."""
    target = os.path.join(DATA, "uwf-zeekdata24")
    if _has_files(target) and not force:
        print("  [skip] datasets/uwf-zeekdata24/ already populated")
        return
    if shutil.which("wget") is None:
        sys.exit("  [error] UWF download needs `wget` (recursive dir scrape). "
                 "Install wget or see datasets/SOURCES.txt for the manual steps.")
    tactics = ["Benign", "Credential_Access", "Reconnaissance", "Initial_Access",
               "Privilege_Escalation", "Persistence", "Defense_Evasion", "Exfiltration"]
    base = "https://datasets.uwf.edu/data/UWF-ZeekData24/csv"
    for tactic in tactics:
        out = os.path.join(target, tactic)
        os.makedirs(out, exist_ok=True)
        print(f"    wget {tactic}/")
        subprocess.run(
            ["wget", "-r", "-np", "-nH", "--cut-dirs=5", "-P", out,
             f"{base}/{tactic}/", "--accept=*.csv", "-q"],
            check=False,
        )
    print("  [ok] UWF-ZeekData24")


def get_ctu_normal(force: bool) -> None:
    """CTU-Normal benign Zeek conn.logs (~49 MB, 13 files)."""
    target = os.path.join(DATA, "ctu-normal")
    if _has_files(target) and not force:
        print("  [skip] datasets/ctu-normal/ already populated")
        return
    os.makedirs(target, exist_ok=True)
    for n in range(20, 33):
        url = f"https://mcfp.felk.cvut.cz/publicDatasets/CTU-Normal-{n}/bro/conn.log"
        dest = os.path.join(target, f"conn-normal-{n}.log")
        try:
            _download(url, dest)
        except Exception as e:  # some CTU-Normal indices may be absent
            print(f"    [warn] CTU-Normal-{n} unavailable: {e}")
    print("  [ok] CTU-Normal")


DATASETS = {
    "iot23":      ("IoT-23 small labeled conn.logs  (~8.7 GB)", get_iot23),
    "ctu13":      ("CTU-13 binetflow                 (~1.9 GB)", get_ctu13),
    "unsw":       ("UNSW-NB15 flow parquet           (~175 MB)", get_unsw),
    "uwf":        ("UWF-ZeekData24 Zeek conn.log CSVs (~21 MB)", get_uwf),
    "ctu_normal": ("CTU-Normal benign Zeek conn.logs  (~49 MB)", get_ctu_normal),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download raw IDS datasets into datasets/.")
    ap.add_argument("--all", action="store_true", help="download every dataset")
    ap.add_argument("--only", metavar="A,B", help="comma-separated subset (see --list)")
    ap.add_argument("--list", action="store_true", help="list dataset keys and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.list or not (args.all or args.only):
        print("Available datasets (key — description):")
        for key, (desc, _fn) in DATASETS.items():
            print(f"  {key:<12} {desc}")
        print("\nAlso: CTU-Malware-Capture is fetched on-demand by preprocess_zeek.py;")
        print("      CICIDS2017 was dropped in v7+ and is not downloaded.")
        print("\nExamples:")
        print("  .venv/bin/python scripts/download_datasets.py --all")
        print("  .venv/bin/python scripts/download_datasets.py --only unsw,ctu_normal")
        return

    keys = list(DATASETS) if args.all else [k.strip() for k in args.only.split(",")]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {', '.join(unknown)}. Run --list.")

    os.makedirs(DATA, exist_ok=True)
    for key in keys:
        desc, fn = DATASETS[key]
        print(f"\n=== {key}: {desc} ===")
        fn(args.force)
    print("\nDone.")


if __name__ == "__main__":
    main()

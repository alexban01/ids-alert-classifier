"""
run_manifest.py — Lightweight experiment provenance (stdlib only, no deps).

The identity of a training run is otherwise smeared across CLI flags, constants
in preprocess_config.py, hyperparams in train.py, and a benchmark JSON. This
module ties them together with self-describing sidecars + a generated ledger:

  <stem>.meta.json     written by preprocess_zeek.py — dataset provenance
                       (git SHA, CLI args, resolved config, counts, content hash)
  <adapter>/run.json   written by train.py — hyperparams + link to the dataset
                       (by content hash); benchmark results appended later
  EXPERIMENTS.md       generated leaderboard, rebuilt from every models/*/run.json

run.json sidecars live inside the gitignored models/ dir (they travel with the
adapter). EXPERIMENTS.md sits at the repo root and is the committed, durable
record — regenerate it any time with `python scripts/experiments.py`.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime

MODELS_DIR     = "models"
EXPERIMENTS_MD = "EXPERIMENTS.md"


# ── Primitives ────────────────────────────────────────────────────────────────
def now():
    """ISO-8601 local timestamp to the second."""
    return datetime.now().isoformat(timespec="seconds")


def git_sha():
    """(short_sha, dirty) for the current checkout, or (None, False) outside git."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
        return sha, dirty
    except Exception:
        return None, False


def git_diff_sha():
    """Short hash of `git diff HEAD`, identifying the exact uncommitted state.

    git_sha() alone is misleading on a dirty tree (the recorded commit may not
    contain the code that actually ran); this pins the working-tree diff so a
    run is auditable even before it's committed. None when clean / outside git.
    """
    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
        )
        return hashlib.sha256(diff).hexdigest()[:12] if diff.strip() else None
    except Exception:
        return None


def detect_reason_from_dataset(train_file, probe=200):
    """True if targets carry a REASON line, False if verdict-only, None if unknown.

    Reads the assistant target of up to `probe` records. This is the AUTHORITATIVE
    source of the reason flag — derived from the data itself, so it's correct even
    on RunPod where only the .jsonl files (not zeek_dataset.meta.json) are uploaded.
    The dataset is homogeneous, so the first record suffices; probing a few guards
    against a stray blank/oddity.
    """
    try:
        seen = 0
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msgs = (json.loads(line).get("messages")) or []
                assistant = next((m.get("content", "") for m in reversed(msgs)
                                  if m.get("role") == "assistant"), None)
                if assistant is None:
                    continue
                seen += 1
                if "REASON:" in assistant:
                    return True
                if seen >= probe:
                    break
        return False if seen else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def file_sha256(path, _bufsize=1 << 20):
    """Streaming SHA-256 of a file's bytes; None if the file is missing."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def dataset_meta_path(train_file):
    """zeek_dataset.jsonl -> zeek_dataset.meta.json"""
    return os.path.splitext(train_file)[0] + ".meta.json"


# ── Writers ───────────────────────────────────────────────────────────────────
def write_dataset_meta(train_file, eval_file, args, config, counts, sources):
    """Write the dataset provenance sidecar next to the train file. Returns it."""
    sha, dirty = git_sha()
    meta = {
        "created":      now(),
        "git_sha":      sha,
        "git_dirty":    dirty,
        "git_diff_sha": git_diff_sha() if dirty else None,
        "args":         args,
        "config":       config,
        "train_file":   train_file,
        "eval_file":    eval_file,
        "counts":       counts,
        "sources":      dict(sources),
        "train_sha256": file_sha256(train_file),
        "eval_sha256":  file_sha256(eval_file),
    }
    write_json(dataset_meta_path(train_file), meta)
    return meta


def write_run_manifest(adapter_dir, *, base_model, target, hyperparams,
                       train_file, eval_loss=None, train_runtime_s=None):
    """Write <adapter_dir>/run.json and regenerate the ledger. Returns the run dict.

    Links the adapter to its dataset by content hash. If the dataset's
    <stem>.meta.json sits beside the train file (i.e. preprocessing ran here),
    the experiment-defining preprocess knobs are surfaced into the run too, so a
    single run.json answers "what got trained, on what, at which settings".
    """
    sha, dirty = git_sha()
    train_sha  = file_sha256(train_file)
    # `reason` is derived from the dataset itself (not the meta sidecar), so it is
    # correct even on RunPod where only the .jsonl files are uploaded — this is
    # what resolve_system_prompt() relies on to serve the matching prompt.
    reason     = detect_reason_from_dataset(train_file)
    dataset    = {"train_file": train_file, "train_sha256": train_sha, "reason": reason}

    meta = read_json(dataset_meta_path(train_file))
    if meta:
        cfg = meta.get("config", {})
        dataset.update({
            "matches_meta":    meta.get("train_sha256") == train_sha,
            "training_factor": cfg.get("TRAINING_FACTOR"),
            "meta_git_sha":    meta.get("git_sha"),
            "counts":          meta.get("counts"),
        })
        if reason is None:                       # dataset probe inconclusive — fall back
            dataset["reason"] = cfg.get("reason")

    run = {
        "created":         now(),
        "git_sha":         sha,
        "git_dirty":       dirty,
        "git_diff_sha":    git_diff_sha() if dirty else None,
        "adapter_dir":     adapter_dir,
        "base_model":      base_model,
        "target":          target,
        "hyperparams":     hyperparams,
        "dataset":         dataset,
        "eval_loss":       eval_loss,
        "train_runtime_s": train_runtime_s,
        "benchmark":       None,
    }
    write_json(os.path.join(adapter_dir, "run.json"), run)
    try:
        regenerate_experiments_md()
    except Exception:
        pass
    return run


def attach_benchmark_result(adapter_dir, result):
    """Append a benchmark block to <adapter_dir>/run.json, then regen the ledger.

    Returns False (silently) if the adapter has no run.json — e.g. a base-model
    row in the benchmark, which has no manifest to update.
    """
    path = os.path.join(adapter_dir, "run.json")
    run  = read_json(path)
    if run is None:
        return False
    run["benchmark"] = result
    write_json(path, run)
    try:
        regenerate_experiments_md()
    except Exception:
        pass
    return True


# ── Ledger ────────────────────────────────────────────────────────────────────
def _fmt(v, spec=None, none="–"):
    if v is None:
        return none
    if spec:
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def _bool(v, none="–"):
    return none if v is None else ("yes" if v else "no")


def regenerate_experiments_md(models_dir=MODELS_DIR, out=EXPERIMENTS_MD):
    """Rebuild EXPERIMENTS.md from every models/*/run.json. Returns row count."""
    rows = []
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            run = read_json(os.path.join(models_dir, name, "run.json"))
            if not run:
                continue
            hp = run.get("hyperparams") or {}
            ds = run.get("dataset")     or {}
            bm = run.get("benchmark")   or {}
            rows.append({
                "run":       name,
                "created":   run.get("created") or "",
                "date":      (run.get("created") or "")[:10],
                "git":       run.get("git_sha") or "–",
                "reason":    ds.get("reason"),
                "epochs":    hp.get("epochs"),
                "pack":      hp.get("packing"),
                "tf":        ds.get("training_factor"),
                "r":         hp.get("lora_r"),
                "ebatch":    hp.get("effective_batch"),
                "eval_loss": run.get("eval_loss"),
                "mcc":       bm.get("mcc"),
                "atk":       bm.get("atk_recall"),
                "fp":        bm.get("ben_recall"),
            })
    rows.sort(key=lambda r: (r["created"], r["run"]), reverse=True)

    header = (
        "| Run | Date | Git | Reason | Epochs | Pack | TF | LoRA r | Eff.batch "
        "| Eval loss¹ | MCC | Atk R | FP R |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines = [
        "# Experiments",
        "",
        "_Generated from `models/*/run.json` — do not edit by hand; rebuild with "
        "`python scripts/experiments.py`._",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['date']} | {r['git']} | {_bool(r['reason'])} "
            f"| {_fmt(r['epochs'])} | {_bool(r['pack'])} | {_fmt(r['tf'])} "
            f"| {_fmt(r['r'])} | {_fmt(r['ebatch'])} | {_fmt(r['eval_loss'], '.4f')} "
            f"| {_fmt(r['mcc'], '+.4f')} | {_fmt(r['atk'], '.1%')} | {_fmt(r['fp'], '.1%')} |"
        )
    if not rows:
        lines.append("| _(no runs yet)_ | | | | | | | | | | | | |")
    lines += [
        "",
        "¹ Eval loss is computed over the full sequence and is comparable only "
        "*within* a run (best-checkpoint selection) — **not** across Reason on/off "
        "rows, which have different target token counts. Use **MCC** for cross-run "
        "comparison.",
        "",
    ]

    with open(out, "w") as f:
        f.write("\n".join(lines))
    return len(rows)

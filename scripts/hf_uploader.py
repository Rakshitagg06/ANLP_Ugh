#!/usr/bin/env python3
"""Upload queued run models to the Hugging Face Hub and delete the local copies.

Runs alongside the training sweep. Each completed run writes
``<queue>/<run_name>/`` (the trainer writes ``<run_name>.partial`` first and
renames it when complete). This process uploads each folder to
``runs/<run_name>/`` in the target repository, verifies that the weights are
present remotely, deletes the local folder, and records the upload. It exits
once the ``STOP`` file exists and the queue is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_CARD = """---
library_name: pytorch
tags: [polarization, semeval-2026-task9, deberta-v3, hierarchy]
---

# POLAR hierarchy models (Team Ugh)

Fine-tuned `microsoft/deberta-v3-base` models for **Structure over Scale:
Hierarchy-Constrained Modeling for English Online Polarization Detection**
(SemEval-2026 Task 9, English POLARDETECT + POLARTYPE).

Each `runs/<model>-fold<k>-seed<s>/` folder contains the best inner-validation
epoch of one outer-fold run:

- `model.safetensors` - FP16 state dict (trained in FP32 with AMP)
- `tokenizer/` - tokenizer files
- `run_metadata.json` - frozen thresholds, best epoch, outer-fold metrics
- `resolved_config.json` - exact training configuration

Conditions: `m1_det`, `m1_type` (independent encoders), `m2` (shared encoder,
unconstrained DET and TYPE heads), `m4_core` (TYPE heads with noisy-OR DET).
M1 (combined) and M3 (post-hoc reconciliation of M2) have no weights of their own.

Load with the project code:

```python
from safetensors.torch import load_file
from polar_hierarchy.models import build_model
model = build_model(resolved_config)            # resolved_config.json
model.load_state_dict(load_file("model.safetensors"))
```
"""


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} | {message}", flush=True)


def ready_folders(queue: Path) -> list[Path]:
    return sorted(
        path
        for path in queue.iterdir()
        if path.is_dir() and not path.name.endswith(".partial")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--uploaded-log", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=True, exist_ok=True, repo_type="model")
    if "README.md" not in api.list_repo_files(args.repo_id):
        api.upload_file(
            path_or_fileobj=MODEL_CARD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Add model card",
        )

    queue = Path(args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    uploaded_log = Path(args.uploaded_log)
    stop_file = queue / "STOP"
    failures: dict[str, int] = {}

    while True:
        pending = ready_folders(queue)
        if not pending:
            if stop_file.exists():
                log("Queue empty and STOP present; exiting")
                return
            time.sleep(args.poll_seconds)
            continue

        folder = pending[0]
        run_name = folder.name
        path_in_repo = f"runs/{run_name}"
        size_mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6
        log(f"Uploading {run_name} ({size_mb:.0f} MB)")
        started = time.time()
        try:
            api.upload_folder(
                folder_path=str(folder),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                commit_message=f"Add {run_name}",
            )
            remote = set(api.list_repo_files(args.repo_id))
            if f"{path_in_repo}/model.safetensors" not in remote:
                raise RuntimeError("model.safetensors missing after upload")
        except Exception as exc:  # Network failures are retried; the folder is kept.
            failures[run_name] = failures.get(run_name, 0) + 1
            wait = min(600, 30 * failures[run_name])
            log(f"Upload of {run_name} failed (attempt {failures[run_name]}): {exc}; retry in {wait}s")
            time.sleep(wait)
            continue

        shutil.rmtree(folder)
        elapsed = time.time() - started
        with uploaded_log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "run_name": run_name,
                        "hf_path": f"{args.repo_id}/{path_in_repo}",
                        "size_mb": round(size_mb, 1),
                        "seconds": round(elapsed, 1),
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
        log(f"Uploaded {run_name} in {elapsed / 60:.1f} min; local copy deleted")


if __name__ == "__main__":
    main()

"""CLI for the AI Daily Global pipeline stages.

    python -m ops.cli status
    python -m ops.cli run <stage> [--force]
    python -m ops.cli retry          # re-run today's failed stages only
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ops import common as C
from ops import stages as S

DEPS = {
    "generate": [],
    "quality": ["generate"],
    "podcast": ["generate", "quality"],
    "publish": ["podcast"],
    "health": [],
    "final": [],
}

ARTIFACTS = {
    "generate": ["data/{d}.raw.json", "daily/{d}.md", "daily/{d}.en.md", "docs/daily/{d}.html"],
    "quality": ["state/{d}.quality.json"],
    "podcast": ["docs/audio/{d}.mp3", "docs/podcast.xml"],
    "publish": [],
    "health": [],
    "final": [],
}

FUNCS = {
    "generate": S.generate,
    "quality": S.quality,
    "podcast": S.podcast,
    "publish": S.publish,
    "health": S.health,
    "final": S.final,
}


def _artifacts(stage: str, date: str) -> list[str]:
    return [a.format(d=date) for a in ARTIFACTS[stage]]


def _ensure_remote() -> None:
    cfg = C.load_config()
    remote = cfg["repo"]["remote"]
    import subprocess

    got = subprocess.run(["git", "remote", "get-url", "origin"], cwd=C.PROJECT,
                         capture_output=True, text=True)
    if got.returncode != 0:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=C.PROJECT, check=True)


def cmd_run(stage: str, force: bool) -> int:
    cfg = C.load_config()
    date = C.today()
    attempts = max(1, int(cfg.get("retry", {}).get("max_attempts", 1)))
    backoff = int(cfg.get("retry", {}).get("backoff_seconds", 300))
    code = C.EXIT_STAGE_FAILED
    for attempt in range(1, attempts + 1):
        with C.FileLock(f"{C.PROJECT.name}-{stage}"):
            code = C.run_stage(stage, DEPS[stage], FUNCS[stage], _artifacts(stage, date),
                               force=force)
        if code == C.EXIT_OK or code == C.EXIT_UPSTREAM_NOT_READY:
            return code
        if attempt < attempts:
            print(f"RETRY {stage} in {backoff}s (attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(backoff)
    return code


def cmd_retry() -> int:
    date = C.today()
    ledger = C.Ledger(date)
    failed = [s for s in ("generate", "quality", "podcast", "publish") if not ledger.is_done(s)]
    if not failed:
        print("nothing to retry")
        return 0
    print(f"retrying failed stages: {failed}")
    for stage in failed:
        code = cmd_run(stage, force=False)
        if code != 0:
            return code
    return 0


def cmd_status() -> int:
    date = C.today()
    ledger = C.Ledger(date)
    print(json.dumps({"project": C.PROJECT.name, "date": date, "stages": ledger.data["stages"]},
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "retry", "status"])
    ap.add_argument("stage", nargs="?", choices=list(DEPS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "retry":
        return cmd_retry()
    if not args.stage:
        ap.error("run needs a stage")
    _ensure_remote()
    return cmd_run(args.stage, args.force)


if __name__ == "__main__":
    sys.exit(main())

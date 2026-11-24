"""Shared ops layer for the AI Daily pipelines.

Design rules (both products use an identical copy of this file so the two
projects stay independent):

* every stage of a day is recorded in ``state/<date>.json`` (atomic write);
* a stage that already finished ``ok`` and whose artifacts still exist is
  skipped on a re-run, so a double fire cannot re-generate, re-upload or
  re-send anything;
* a stage refuses to run when its upstream is not ``ok`` — no publishing of
  a stale report;
* failures keep the artifacts that did succeed and record the error, so the
  retry can re-run only the failed stage;
* nothing here prints or stores credentials: the API key is read at runtime
  from the Hermes env file and only exported into the child process env.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BEIJING = timezone(timedelta(hours=8))
PROJECT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT / "config" / "pipeline.json"
STATE_DIR = PROJECT / "state"
LOG_DIR = PROJECT / "logs"

EXIT_OK = 0
EXIT_STAGE_FAILED = 1
EXIT_UPSTREAM_NOT_READY = 3
EXIT_ALREADY_DONE = 0


def today() -> str:
    return datetime.now(BEIJING).strftime("%Y-%m-%d")


def now_iso() -> str:
    return datetime.now(BEIJING).isoformat(timespec="seconds")


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.setdefault("project", PROJECT.name)
    return cfg


# --------------------------------------------------------------------------- #
# per-day ledger
# --------------------------------------------------------------------------- #
class Ledger:
    """Atomic per-day stage ledger."""

    def __init__(self, date: str):
        self.date = date
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = STATE_DIR / f"{date}.json"
        self.data = self._read()

    def _read(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                broken = self.path.with_suffix(".broken.json")
                self.path.replace(broken)
        return {"date": self.date, "project": PROJECT.name, "stages": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def stage(self, name: str) -> dict:
        return self.data["stages"].setdefault(
            name, {"status": "pending", "attempts": 0, "artifacts": {}, "error": None}
        )

    def status(self, name: str) -> str:
        return self.data["stages"].get(name, {}).get("status", "pending")

    # -- guards -------------------------------------------------------------
    def missing_upstream(self, deps: list[str]) -> list[str]:
        return [d for d in deps if self.status(d) != "ok"]

    def artifacts_present(self, stage: str) -> bool:
        files = self.stage(stage).get("artifacts") or {}
        if not files:
            # stages that only assert on other stages' files (publish/final)
            return True
        for rel, meta in files.items():
            p = PROJECT / rel
            if not p.exists() or p.stat().st_size == 0:
                return False
            size = (meta or {}).get("size")
            if size and p.stat().st_size != size:
                return False
        return True

    def is_done(self, stage: str) -> bool:
        return self.status(stage) == "ok" and self.artifacts_present(stage)


# --------------------------------------------------------------------------- #
# single-flight lock
# --------------------------------------------------------------------------- #
class FileLock:
    def __init__(self, name: str, stale_minutes: int = 120):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = STATE_DIR / f".{name}.lock"
        self.stale = stale_minutes * 60
        self.handle = None

    def __enter__(self):
        self.handle = open(self.path, "w")
        handle = self.handle
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                age = time.time() - self.path.stat().st_mtime
                if age > self.stale:
                    # holder died long ago; take over
                    fcntl.flock(handle, fcntl.LOCK_EX)
                else:
                    handle.close()
                    self.handle = None
                    raise SystemExit(f"SKIP already_running (lock {self.path.name}, {int(age)}s)")
            else:
                raise
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()} {now_iso()}\n")
        handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
            except Exception:
                pass
        return False


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def logger(stage: str, date: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"{date}-{stage}.log"

    def log(msg: str) -> None:
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log, logfile


# --------------------------------------------------------------------------- #
# stage runner
# --------------------------------------------------------------------------- #
def run_stage(name: str, deps: list[str], fn, artifacts: list[str], force: bool = False) -> int:
    """Run one stage idempotently. ``fn(log) -> dict`` returns extra state."""
    date = today()
    log, logfile = logger(name, date)
    ledger = Ledger(date)
    project = PROJECT.name
    log(f"== {project}/{name} {date} ==")

    if not force and ledger.is_done(name):
        log(f"SKIP {name} already ok; artifacts intact")
        return EXIT_ALREADY_DONE

    missing = ledger.missing_upstream(deps)
    if missing:
        detail = ", ".join(f"{d}={ledger.status(d)}" for d in missing)
        log(f"BLOCKED upstream not ready: {detail}")
        ledger.stage(name).update(
            status="blocked", error=f"upstream not ready: {detail}", checked_at=now_iso()
        )
        ledger.save()
        return EXIT_UPSTREAM_NOT_READY

    st = ledger.stage(name)
    st.update(status="running", attempts=st.get("attempts", 0) + 1, started_at=now_iso(), error=None)
    ledger.save()
    try:
        extra = fn(log) or {}
    except SystemExit as exc:
        st.update(status="failed", finished_at=now_iso(), error=f"SystemExit {exc.code}")
        ledger.save()
        log(f"FAIL {name}: {exc}")
        return EXIT_STAGE_FAILED
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded
        import traceback

        tb = traceback.format_exc()
        st.update(status="failed", finished_at=now_iso(), error=f"{type(exc).__name__}: {exc}")
        ledger.save()
        log(f"FAIL {name}: {type(exc).__name__}: {exc}")
        log(tb.strip().splitlines()[-1])
        return EXIT_STAGE_FAILED

    files = {}
    for rel in artifacts:
        p = PROJECT / rel
        if p.exists() and p.stat().st_size > 0:
            files[rel] = {"size": p.stat().st_size}
    absent = [a for a in artifacts if a not in files]
    if absent:
        st.update(status="failed", finished_at=now_iso(), error=f"missing artifacts: {absent}")
        ledger.save()
        log(f"FAIL {name}: missing artifacts {absent}")
        return EXIT_STAGE_FAILED

    st.update(status="ok", finished_at=now_iso(), artifacts=files, error=None)
    st.update(extra)
    ledger.save()
    log(f"OK {name} artifacts={len(files)} log={logfile.name}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# env / credentials
# --------------------------------------------------------------------------- #
def hermes_env(key: str) -> str:
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def child_env(cfg: dict, *, proxy: bool = True) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    env["TZ"] = "Asia/Shanghai"
    model = cfg.get("model", {})
    api_key = hermes_env(model.get("api_key_env", "DEEPSEEK_API_KEY"))
    if not api_key:
        raise RuntimeError(f"missing API key env {model.get('api_key_env')}")
    env["API_KEY"] = api_key
    env.setdefault("DEEPSEEK_API_KEY", api_key)
    env["API_BASE_URL"] = model.get("base_url", "https://api.deepseek.com")
    env["API_MODEL"] = model.get("name", "deepseek-chat")
    net = cfg.get("network", {})
    if proxy and net.get("proxy"):
        env["HTTPS_PROXY"] = env["https_proxy"] = net["proxy"]
        env["HTTP_PROXY"] = env["http_proxy"] = net["proxy"]
        env["NO_PROXY"] = env["no_proxy"] = net.get("no_proxy", "127.0.0.1,localhost")
    elif not proxy:
        for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            env.pop(k, None)
        env["NO_PROXY"] = env["no_proxy"] = "*"
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    return env


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None,
            log=None, timeout: int = 3600) -> subprocess.CompletedProcess:
    log = log or (lambda *_: None)
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd or PROJECT), env=env, capture_output=True, text=True, timeout=timeout
    )
    for stream in (proc.stdout, proc.stderr):
        for line in (stream or "").splitlines():
            log("  " + line)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {proc.returncode}")
    return proc


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
def http_status(url: str, proxy: str | None = None, method: str = "GET",
                timeout: int = 25) -> tuple[int, int]:
    """Return (status, bytes). 0 means the request itself failed."""
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, method=method, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "Cache-Control": "no-cache",
    })
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read() if method != "HEAD" else b""
            return resp.status, len(body)
    except urllib.error.HTTPError as exc:
        return exc.code, 0
    except Exception:
        return 0, 0


def http_json(url: str, proxy: str | None = None, timeout: int = 30) -> dict:
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy
                else urllib.request.ProxyHandler({})]
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-daily-ops/1.0"})
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# notify
# --------------------------------------------------------------------------- #
def notify(cfg: dict, title: str, body_md: str, attachments: list[str] | None = None) -> bool:
    """Send through the user's configured Telegram channel via `hermes send`."""
    target_file = Path(cfg.get("notify", {}).get("target_file", "")).expanduser()
    if not target_file or not target_file.exists():
        return False
    target = target_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"telegram:-?\d+(:\d+)?", target):
        return False
    hermes = Path.home() / ".local" / "bin" / "hermes"
    if not hermes.exists():
        return False
    text = f"{title}\n\n{body_md}"
    for path in attachments or []:
        text += f"\nMEDIA:{path}"
    env = dict(os.environ)
    env["TELEGRAM_PROXY"] = cfg.get("notify", {}).get("proxy", "")
    proc = subprocess.run([str(hermes), "send", "--to", target, text],
                          capture_output=True, text=True, env=env, timeout=180)
    return proc.returncode == 0

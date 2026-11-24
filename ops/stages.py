"""Stage implementations for AI Daily Global (综合版)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path

from ops import common as C

PY = C.PROJECT / ".venv" / "bin" / "python"
SCRIPTS = C.PROJECT / "scripts"


def _cfg() -> dict:
    return C.load_config()


def _script_env(cfg: dict) -> dict:
    env = C.child_env(cfg, proxy=True)
    env.setdefault("PYTHONPATH", str(SCRIPTS))
    return env


def _md(date: str) -> Path:
    return C.PROJECT / "daily" / f"{date}.md"


def _md_en(date: str) -> Path:
    return C.PROJECT / "daily" / f"{date}.en.md"


def _page(date: str) -> Path:
    return C.PROJECT / "docs" / "daily" / f"{date}.html"


def _mp3(date: str) -> Path:
    return C.PROJECT / "docs" / "audio" / f"{date}.mp3"


def _run_main(stage: str, date: str, log, cfg: dict, force: bool = False) -> None:
    cmd = [str(PY), str(SCRIPTS / "main.py"), "--stage", stage, "--date", date]
    if force:
        cmd.append("--force")
    C.run_cmd(cmd, cwd=SCRIPTS, env=_script_env(cfg), log=log, timeout=3600)


# --------------------------------------------------------------------------- #
# 1. generate — 04:00
# --------------------------------------------------------------------------- #
def generate(log) -> dict:
    cfg = _cfg()
    date = C.today()
    if not PY.exists():
        raise RuntimeError(f"missing venv python {PY}")

    _run_main("fetch", date, log, cfg)
    raw = C.PROJECT / "data" / f"{date}.raw.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    total = sum(counts.values())
    if total < 5:
        # Zero (or near-zero) collection is a failure, not an empty success.
        raise RuntimeError(f"collected only {total} items — refusing to compose")
    log(f"collected {total} items {counts}")

    _run_main("compose", date, log, cfg)
    _run_main("site", date, log, cfg)

    health = payload.get("health") or []
    failed = [h for h in health if h.get("status") != "ok"]
    return {
        "counts": counts,
        "sources_total": len(health),
        "sources_failed": [f"{h['source']}: {h.get('note')}" for h in failed][:20],
        "total_items": total,
    }


# --------------------------------------------------------------------------- #
# 2. quality — 04:30
# --------------------------------------------------------------------------- #
def _titles(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^-\s+\*\*\[?(.+?)\]?\*\*", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _links(text: str) -> list[str]:
    return re.findall(r"\]\((https?://[^)\s]+)\)", text)


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def quality(log) -> dict:
    cfg = _cfg()
    q = cfg["quality"]
    date = C.today()
    problems: list[str] = []

    md, md_en, page = _md(date), _md_en(date), _page(date)
    for path, label in ((md, "中文版"), (md_en, "英文版")):
        if not path.exists():
            problems.append(f"{label}缺失 ({path.name})")
        elif len(path.read_text(encoding="utf-8")) < q["min_md_chars"]:
            problems.append(f"{label}过短 ({path.stat().st_size} B)")

    if not page.exists():
        problems.append("网页未生成")
    else:
        html = page.read_text(encoding="utf-8")
        size = len(html.encode())
        if size < q["min_html_bytes"]:
            problems.append(f"网页过小 ({size} B)")
        for marker in ("市场局势", "赛道深挖", "今日观察"):
            if marker not in html:
                problems.append(f"网页缺少分区 {marker}")

    if not md.exists():
        return {"passed": False, "problems": problems, "checked_at": C.now_iso()}

    text = md.read_text(encoding="utf-8")
    titles = _titles(text)
    if len(titles) < 6:
        problems.append(f"资讯条目过少 ({len(titles)})")

    # cross-day dedupe against the previous 5 editions
    prev_days = sorted(p for p in (C.PROJECT / "daily").glob("*.md")
                       if not p.name.endswith(".en.md") and p.stem < date)[-q["dedupe_days"]:]
    repeats = []
    for prev in prev_days:
        old = _titles(prev.read_text(encoding="utf-8"))
        for t in titles:
            for o in old:
                if _similar(t, o) > 0.82:
                    repeats.append({"title": t, "dup_of": prev.stem, "old": o})
                    break
            else:
                continue
            break
    ratio = len(repeats) / max(len(titles), 1)
    log(f"dedupe: {len(titles)} titles, {len(repeats)} repeats over {len(prev_days)} days ({ratio:.0%})")
    if ratio > q["max_title_overlap_ratio"]:
        problems.append(f"跨日重复率过高 {ratio:.0%} ({len(repeats)}/{len(titles)})")

    # source reachability sample — a 403/404-heavy run means the links are wrong
    links = list(dict.fromkeys(_links(text)))[: q["probe_sample"]]
    probe = []
    for url in links:
        status, _ = C.http_status(url, proxy=cfg["network"]["proxy"], method="HEAD")
        probe.append({"url": url, "status": status})
    ok = sum(1 for p in probe if 200 <= p["status"] < 400)
    ok_ratio = ok / len(probe) if probe else 1.0
    log(f"link probe: {ok}/{len(probe)} ok ({ok_ratio:.0%})")
    if probe and ok_ratio < q["min_probe_ok_ratio"]:
        problems.append(f"来源链接可用率过低 {ok_ratio:.0%}")

    report = {
        "passed": not problems,
        "problems": problems,
        "checked_at": C.now_iso(),
        "titles": len(titles),
        "repeat_ratio": round(ratio, 3),
        "repeats": repeats[:20],
        "probe": probe,
        "html_bytes": page.stat().st_size if page.exists() else 0,
    }
    (C.STATE_DIR / f"{date}.quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("; ".join(problems))
    return report


# --------------------------------------------------------------------------- #
# 3. podcast — 05:00
# --------------------------------------------------------------------------- #
def podcast(log) -> dict:
    cfg = _cfg()
    date = C.today()
    _run_main("audio", date, log, cfg)
    mp3 = _mp3(date)
    if not mp3.exists() or mp3.stat().st_size < cfg["audio"]["min_bytes"]:
        raise RuntimeError(f"音频不可用 {mp3}")
    # duration sanity via mutagen (the page/podcast needs a real playable file)
    try:
        from mutagen.mp3 import MP3

        secs = int(MP3(str(mp3)).info.length)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"MP3 无法解析: {exc}") from exc
    if secs < 300:
        raise RuntimeError(f"音频过短 {secs}s")
    log(f"mp3 {mp3.stat().st_size} B, {secs}s")

    _run_main("site", date, log, cfg)
    rss = C.PROJECT / "docs" / "podcast.xml"
    if not rss.exists() or date not in rss.read_text(encoding="utf-8"):
        raise RuntimeError("播客 RSS 未包含当天单集")
    log("rss contains today's episode")

    email_code = 0
    try:
        _run_main("email", date, log, cfg)
    except RuntimeError as exc:
        email_code = 4
        log(f"email failed (non-fatal): {exc}")
    return {"seconds": secs, "bytes": mp3.stat().st_size, "email": "ok" if email_code == 0 else "failed"}


# --------------------------------------------------------------------------- #
# 4. publish — 05:00
# --------------------------------------------------------------------------- #
def _git(*args: str, log, check: bool = True) -> subprocess.CompletedProcess:
    return C.run_cmd(["git", *args], cwd=C.PROJECT, log=log) if check else subprocess.run(
        ["git", *args], cwd=C.PROJECT, capture_output=True, text=True)


def publish(log) -> dict:
    cfg = _cfg()
    date = C.today()
    repo = cfg["repo"]
    paths = ["daily", "data", "docs", "reports", "scripts", "tests", "ops", "config"]
    _git("add", *[p for p in paths if (C.PROJECT / p).exists()], log=log)
    staged = _git("diff", "--cached", "--quiet", log=log, check=False)
    if staged.returncode != 0:
        _git("-c", "user.name=AI Daily Bot", "-c", "user.email=bot@local",
             "commit", "-m", f"📰 Daily digest for {date}", log=log)
    else:
        log("nothing to commit")

    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=C.PROJECT,
                            capture_output=True, text=True)
    if remote.returncode != 0:
        _git("remote", "add", "origin", repo["remote"], log=log)

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0",
               https_proxy=cfg["network"]["proxy"], http_proxy=cfg["network"]["proxy"])
    push = subprocess.run(["git", "push", "origin", f"HEAD:{repo['branch']}"], cwd=C.PROJECT,
                          capture_output=True, text=True, env=env, timeout=300)
    log("$ git push origin HEAD:" + repo["branch"])
    for line in (push.stderr or "").splitlines():
        log("  " + line)
    if push.returncode != 0:
        if "Authentication" in push.stderr or "could not read Username" in push.stderr:
            raise RuntimeError(
                "GitHub 推送需要授权：在本机执行一次 `gh auth login`（或配置 PAT）。"
                "内容已在本地生成并提交，重新运行该阶段即可继续。"
            )
        raise RuntimeError(f"git push failed: {push.stderr.strip()[:300]}")

    # verify what the public actually serves
    base = repo["pages_base"]
    page_url = f"{base}/daily/{date}.html"
    rss_url = repo["rss"]
    mp3_url = f"{base}/audio/{date}.mp3"
    deadline = time.time() + 600
    state = {}
    while time.time() < deadline:
        page_status, page_bytes = C.http_status(page_url, proxy=cfg["network"]["proxy"])
        rss_body = ""
        status_rss, _ = C.http_status(rss_url, proxy=cfg["network"]["proxy"])
        if status_rss == 200:
            rss_body = _fetch(rss_url, cfg["network"]["proxy"])
        mp3_status, mp3_bytes = C.http_status(mp3_url, proxy=cfg["network"]["proxy"], method="HEAD")
        state = {"page": page_status, "page_bytes": page_bytes, "rss": status_rss,
                 "mp3": mp3_status, "mp3_bytes": mp3_bytes,
                 "rss_has_episode": date in rss_body}
        if page_status == 200 and page_bytes > cfg["quality"]["min_html_bytes"] and status_rss == 200 \
                and date in rss_body and mp3_status == 200:
            break
        log(f"waiting for Pages propagation {state}")
        time.sleep(30)
    log(f"public verify {state}")
    if not (state.get("page") == 200 and state.get("mp3") == 200 and state.get("rss_has_episode")):
        raise RuntimeError(f"公网校验未通过: {state}")
    return state


def _fetch(url: str, proxy: str) -> str:
    import urllib.request

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    req = urllib.request.Request(url, headers={"User-Agent": "ai-daily-ops/1.0",
                                               "Cache-Control": "no-cache"})
    try:
        with opener.open(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "ignore")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 5. health — 06:15
# --------------------------------------------------------------------------- #
def health(log) -> dict:
    cfg = _cfg()
    date = C.today()
    ledger = C.Ledger(date)
    checks: dict[str, object] = {}

    gw = subprocess.run([str(Path.home() / ".local/bin/hermes"), "gateway", "status"],
                        capture_output=True, text=True)
    checks["gateway"] = "running" if gw.returncode == 0 and "running" in gw.stdout.lower() else "down"

    df = subprocess.run(["df", "-g", "/"], capture_output=True, text=True)
    try:
        avail = int(df.stdout.splitlines()[1].split()[3])
    except (IndexError, ValueError):
        avail = -1
    checks["disk_free_gb"] = avail
    checks["disk_ok"] = avail > 10

    st, _ = C.http_status("https://github.com", proxy=cfg["network"]["proxy"], method="HEAD")
    checks["network"] = st

    push_probe = subprocess.run(["git", "ls-remote", "--heads", "origin"], cwd=C.PROJECT,
                               capture_output=True, text=True,
                               env=dict(os.environ, GIT_TERMINAL_PROMPT="0",
                                        https_proxy=cfg["network"]["proxy"]))
    checks["git_remote_reachable"] = push_probe.returncode == 0

    checks["stages"] = {k: v.get("status") for k, v in ledger.data["stages"].items()}
    log(f"health {json.dumps(checks, ensure_ascii=False)}")

    page = _page(date)
    if not page.exists():
        raise RuntimeError("today's page missing at 06:15")
    return checks


# --------------------------------------------------------------------------- #
# 6. final — 07:00
# --------------------------------------------------------------------------- #
def final(log) -> dict:
    cfg = _cfg()
    date = C.today()
    ledger = C.Ledger(date)
    base = cfg["repo"]["pages_base"]
    report = {
        "date": date,
        "中文版": _md(date).exists(),
        "英文版": _md_en(date).exists(),
        "网页": f"{base}/daily/{date}.html",
        "音频": f"{base}/audio/{date}.mp3",
        "RSS": cfg["repo"]["rss"],
        "stages": {k: v.get("status") for k, v in ledger.data["stages"].items()},
    }
    page_status, _ = C.http_status(report["网页"], proxy=cfg["network"]["proxy"])
    mp3_status, _ = C.http_status(report["音频"], proxy=cfg["network"]["proxy"], method="HEAD")
    report["公共可访问"] = {"page": page_status, "mp3": mp3_status}
    log(json.dumps(report, ensure_ascii=False))

    failed = [k for k, v in report["stages"].items() if v not in ("ok", "skipped")]
    ok = page_status == 200 and mp3_status == 200 and not failed
    if not ok:
        body = [f"❌ 综合版 {date} 未全部完成"]
        body.append(f"阶段状态: {report['stages']}")
        if failed:
            errs = [f"{k}: {ledger.stage(k).get('error')}" for k in failed]
            body.append("错误: " + "; ".join(e for e in errs if e))
        body.append(f"网页 {page_status} · 音频 {mp3_status}")
        C.notify(cfg, "AI Daily Global 异常", "\n".join(body))
        raise RuntimeError("delivery incomplete: " + ", ".join(failed or ["public urls"]))
    C.notify(cfg, f"✅ AI Daily Global {date}", "\n".join([
        f"网页 {report['网页']}",
        f"音频 {report['音频']}",
        f"RSS {report['RSS']}",
    ]))
    return report

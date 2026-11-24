"""
AI Daily Digest — staged entry point.

The digest is produced in separate stages so a failure can be retried on its
own (a broken TTS run must not re-spend on summarisation, and a failed email
must not re-render the page):

    python main.py --stage fetch     # collect + cross-day dedupe → data/<date>.raw.json
    python main.py --stage compose   # bilingual summary      → daily/<date>.md[,en.md]
    python main.py --stage audio     # voice broadcast        → docs/audio/<date>.mp3
    python main.py --stage site      # static site + podcast  → docs/
    python main.py --stage email     # Buttondown (idempotent per day)
    python main.py --stage all       # the legacy full sequence

Exit codes: 0 ok · 2 nothing fetched · 3 audio failed · 4 email failed ·
5 a required upstream artifact is missing.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

import sources
from sources import fetch_all, normalize_url
from summarize import summarize
from generate_site import generate_site
from audio import generate_audio, prune_old_audio

ROOT = Path(__file__).resolve().parent.parent
# Episodes are kept for a year instead of the old 15: pruning that far back used
# to kill the audio of every page older than two weeks.
AUDIO_KEEP = int(os.environ.get("AUDIO_KEEP", "365"))


def _date_str() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


def send_to_buttondown(subject: str, markdown: str) -> bool:
    """Publish the digest as a Buttondown email. Returns True when it is out."""
    api_key = os.environ.get("BUTTONDOWN_API_KEY")
    if not api_key:
        print("[Buttondown] BUTTONDOWN_API_KEY not set, skipping.")
        return True  # not configured is not a failure

    resp = requests.post(
        "https://api.buttondown.email/v1/emails",
        headers={
            "Authorization": f"Token {api_key}",
            "X-Buttondown-Live-Dangerously": "true",
        },
        json={
            "subject": subject,
            "body": markdown,
            "status": "about_to_send",  # queues for immediate send
            "email_type": "public",     # visible in archive
        },
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"[Buttondown] Sent! Email id: {resp.json().get('id')}")
        return True
    print(f"[Buttondown] Failed {resp.status_code}: {resp.text[:200]}")
    return False


def load_recent_reported(daily_dir: Path, today: str, days: int = 5):
    """Source URLs and titles from the last `days` published digests (zh only)."""
    urls, titles = set(), []
    files = sorted(p for p in daily_dir.glob("*.md")
                   if not p.name.endswith(".en.md") and p.stem < today)[-days:]
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("### "):   # 今日观察 / 商业机会 / 研究建议: no news items below
                break
            m = re.match(r"^-\s+\*\*\[?(.+?)\]?\*\*", line)
            if m:
                titles.append(m.group(1).strip())
            if re.match(r"^\s*-\s+来源[：:]", line):
                for u in re.findall(r"\]\(([^)\s]+)\)", line):
                    urls.add(normalize_url(u))
    return urls, titles


def drop_already_reported(data: dict, urls: set) -> int:
    removed = 0
    for key, items in data.items():
        # Page-watch items reuse the same URL every time the page changes, so they
        # are never dropped as "already reported".
        kept = [it for it in items
                if it.get("metadata", {}).get("watch") or it.get("normalized_url") not in urls]
        removed += len(items) - len(kept)
        data[key] = kept
    return removed


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_fetch(date_str: str, force: bool) -> int:
    raw_file = ROOT / "data" / f"{date_str}.raw.json"
    if raw_file.exists() and not force:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        total = sum(payload.get("counts", {}).values())
        if total:
            print(f"[fetch] reuse {raw_file.name} ({total} items)")
            return 0
        print(f"[fetch] {raw_file.name} holds zero items; refetching")

    print(f"=== 抓取 {date_str} ===")
    data = fetch_all()
    total = sum(len(v) for v in data.values())
    print(f"Total items fetched: {total}")
    print("[Source health]")
    for row in sources.LAST_HEALTH:
        mark = "OK  " if row["status"] == "ok" else "FAIL"
        print(f"  {mark} [{row['type']}] {row['source']}: {row['note']}")

    daily_dir = ROOT / "daily"
    recent_urls, recent_titles = load_recent_reported(daily_dir, date_str)
    removed = drop_already_reported(data, recent_urls)
    print(f"Cross-day dedupe: removed {removed} items already reported "
          f"(checked {len(recent_titles)} recent titles)")

    raw_file.parent.mkdir(exist_ok=True)
    raw_file.write_text(json.dumps({
        "date": date_str,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "counts": {k: len(v) for k, v in data.items()},
        "health": sources.LAST_HEALTH,
        "recent_titles": recent_titles,
        "items": data,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fetch] saved {raw_file}")

    if total == 0:
        # Zero collected news is a hard failure, never a "successful" empty day.
        print("[fetch] FAILED: no items fetched from any source")
        return 2
    healthy = sum(1 for r in sources.LAST_HEALTH if r["status"] == "ok")
    print(f"[fetch] sources healthy {healthy}/{len(sources.LAST_HEALTH)}")
    return 0


def stage_compose(date_str: str, force: bool) -> int:
    md = ROOT / "daily" / f"{date_str}.md"
    md_en = ROOT / "daily" / f"{date_str}.en.md"
    if md.exists() and md.stat().st_size > 2000 and not force:
        print(f"[compose] reuse {md.name} (+ {'en' if md_en.exists() else 'no en'})")
        return 0
    raw_file = ROOT / "data" / f"{date_str}.raw.json"
    if not raw_file.exists():
        print(f"[compose] missing {raw_file}")
        return 5
    payload = json.loads(raw_file.read_text(encoding="utf-8"))
    data = payload["items"]
    recent_titles = payload.get("recent_titles") or []
    print(f"[compose] summarising {sum(len(v) for v in data.values())} items")
    markdown, markdown_en = summarize(data, date_str, recent_titles=recent_titles)
    (ROOT / "daily").mkdir(exist_ok=True)
    md.write_text(markdown, encoding="utf-8")
    print(f"[compose] wrote {md} ({md.stat().st_size} B)")
    if markdown_en:
        md_en.write_text(markdown_en, encoding="utf-8")
        print(f"[compose] wrote {md_en} ({md_en.stat().st_size} B)")
    else:
        print("[compose] WARNING no English version produced")
    return 0


def stage_audio(date_str: str, force: bool) -> int:
    md = ROOT / "daily" / f"{date_str}.md"
    if not md.exists():
        print(f"[audio] missing {md}")
        return 5
    audio_dir = ROOT / "docs" / "audio"
    out = audio_dir / f"{date_str}.mp3"
    if out.exists() and out.stat().st_size > 200_000 and not force:
        print(f"[audio] reuse {out.name} ({out.stat().st_size} B)")
        return 0
    print("[audio] synthesising voice broadcast")
    try:
        generate_audio(md.read_text(encoding="utf-8"), date_str, audio_dir)
    except Exception as exc:  # noqa: BLE001 — the failure must surface, not vanish
        print(f"[audio] FAILED: {type(exc).__name__}: {exc}")
        return 3
    if not out.exists() or out.stat().st_size < 200_000:
        print(f"[audio] FAILED: {out} missing or too small")
        return 3
    print(f"[audio] wrote {out} ({out.stat().st_size} B)")
    prune_old_audio(audio_dir, keep=AUDIO_KEEP)
    return 0


def stage_site(date_str: str, force: bool) -> int:
    print("[site] rebuilding docs/")
    generate_site(root=ROOT)
    page = ROOT / "docs" / "daily" / f"{date_str}.html"
    if not page.exists():
        print(f"[site] FAILED: {page} missing")
        return 5
    print(f"[site] {page} ({page.stat().st_size} B)")
    return 0


def stage_email(date_str: str, force: bool) -> int:
    marker = ROOT / "state" / f"{date_str}.email_sent"
    if marker.exists() and not force:
        print(f"[email] already sent today ({marker.read_text().strip()})")
        return 0
    md = ROOT / "daily" / f"{date_str}.md"
    if not md.exists():
        print(f"[email] missing {md}")
        return 5
    ok = send_to_buttondown(f"Andy 的 AI 早报 · {date_str}", md.read_text(encoding="utf-8"))
    if not ok:
        return 4
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                      encoding="utf-8")
    return 0


STAGES = {
    "fetch": stage_fetch,
    "compose": stage_compose,
    "audio": stage_audio,
    "site": stage_site,
    "email": stage_email,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    ap.add_argument("--date", help="override date (default: today, Beijing)")
    ap.add_argument("--force", action="store_true", help="ignore existing artifacts")
    args = ap.parse_args()

    date_str = args.date or _date_str()
    order = ["fetch", "compose", "audio", "site", "email"] if args.stage == "all" else [args.stage]
    worst = 0
    for name in order:
        code = STAGES[name](date_str, args.force)
        if code != 0:
            print(f"[{name}] exited {code}; stopping")
            return code
        worst = max(worst, code)
    print(f"Done ({date_str}).")
    return worst


if __name__ == "__main__":
    sys.exit(main())

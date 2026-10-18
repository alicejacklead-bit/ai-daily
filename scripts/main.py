"""
AI Daily Digest - Main entry point.
Fetches AI news from multiple sources, summarizes with AI, and saves outputs.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

import re

from sources import fetch_all, normalize_url
from summarize import summarize
from generate_site import generate_site
from audio import generate_audio, prune_old_audio


def send_to_buttondown(subject: str, markdown: str) -> None:
    """Publish the digest as a Buttondown email (sends to all subscribers)."""
    api_key = os.environ.get("BUTTONDOWN_API_KEY")
    if not api_key:
        print("[Buttondown] BUTTONDOWN_API_KEY not set, skipping.")
        return

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
            "email_type": "public",    # visible in archive
        },
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"[Buttondown] Sent! Email id: {resp.json().get('id')}")
    else:
        print(f"[Buttondown] Failed {resp.status_code}: {resp.text[:200]}")


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
        kept = [it for it in items if it.get("normalized_url") not in urls]
        removed += len(items) - len(kept)
        data[key] = kept
    return removed


def main():
    # Use Beijing time for date
    beijing_tz = timezone(timedelta(hours=8))
    today = datetime.now(beijing_tz)
    date_str = today.strftime("%Y-%m-%d")

    print(f"=== Andy 的 AI 早报 for {date_str} ===\n")

    # Step 1: Fetch data from all sources
    print("[Step 1] Fetching data from sources...")
    data = fetch_all()

    total = sum(len(v) for v in data.values())
    print(f"\nTotal items fetched: {total}")

    if total == 0:
        print("No items fetched. Exiting.")
        sys.exit(0)

    # Cross-day dedupe: drop links already used in the last few digests,
    # and hand the model recent titles so it skips re-reports of the same story.
    daily_dir = Path(__file__).parent.parent / "daily"
    recent_urls, recent_titles = load_recent_reported(daily_dir, date_str)
    removed = drop_already_reported(data, recent_urls)
    print(f"Cross-day dedupe: removed {removed} items already reported "
          f"(checked {len(recent_titles)} recent titles)")

    # Step 2: Save raw fetched data for traceability
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    raw_file = data_dir / f"{date_str}.raw.json"
    raw_payload = {
        "date": date_str,
        "generated_at": today.isoformat(),
        "counts": {key: len(value) for key, value in data.items()},
        "items": data,
    }
    raw_file.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[Step 2] Saved raw data to {raw_file}")

    # Step 3: AI summarization (bilingual)
    print("\n[Step 3] Generating AI summary...")
    markdown, markdown_en = summarize(data, date_str, recent_titles=recent_titles)

    # Step 4: Save daily markdown (zh + en)
    output_dir = Path(__file__).parent.parent / "daily"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{date_str}.md"

    output_file.write_text(markdown, encoding="utf-8")
    print(f"\n[Step 4] Saved to {output_file}")

    if markdown_en:
        output_file_en = output_dir / f"{date_str}.en.md"
        output_file_en.write_text(markdown_en, encoding="utf-8")
        print(f"          Saved English to {output_file_en}")
    else:
        print("          No English version generated.")

    # Step 5: Synthesize voice broadcast (MP3 for commute listening)
    print("\n[Step 5] Generating voice broadcast...")
    audio_dir = Path(__file__).parent.parent / "docs" / "audio"
    try:
        generate_audio(markdown, date_str, audio_dir)
        prune_old_audio(audio_dir, keep=15)  # keep ~half a month of episodes
    except Exception as e:
        print(f"  Audio generation skipped: {e}")

    # Step 6: Rebuild static site (picks up today's audio + podcast feed)
    print("\n[Step 6] Rebuilding static site...")
    generate_site(root=Path(__file__).parent.parent)
    print("  Site rebuilt → docs/")

    # Step 7: Send to Buttondown subscribers
    print("\n[Step 7] Sending to Buttondown subscribers...")
    subject = f"Andy 的 AI 早报 · {date_str}"
    send_to_buttondown(subject, markdown)

    print("Done!")


if __name__ == "__main__":
    main()

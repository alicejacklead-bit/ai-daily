"""
Page watch: snapshot official pricing / changelog pages and report only real changes.

Each run fetches every page in WATCH_PAGES, extracts the visible text, and
compares it with the snapshot saved last run (data/snapshots/<slug>.txt, which
the workflow commits). Rules:
- first run for a page: save a baseline, report nothing;
- small change: emit one news item listing the added / removed lines, so the
  model can quote concrete before/after numbers;
- huge change (layout redesign): re-baseline silently and log it, no report;
- fetch failure or near-empty text (likely JS-rendered): logged in the health
  report, never blocks the run.
"""

import json
import re
from pathlib import Path

import requests

from sources import BROWSER_HEADERS, make_item

SNAPSHOT_DIR = Path(__file__).parent.parent / "data" / "snapshots"

# (slug, display name, url or [fallback urls], kind)
# kind: "html" = text diff of the page; "openrouter" = filtered JSON model catalog.
# Only public industry pages — never add a customer's site here (repo is public).
WATCH_PAGES = [
    # ── 上游供给 ──────────────────────────────────────────────────────────
    ("byteplus-modelark-pricing", "BytePlus ModelArk 价格",
     "https://docs.byteplus.com/en/docs/ModelArk/model-service-pricing", "html"),
    ("volcengine-ark-pricing", "火山方舟 模型价格",
     "https://www.volcengine.com/docs/82379/1544106", "html"),
    ("minimax-pricing", "MiniMax 定价",
     "https://platform.minimax.io/docs/guides/pricing", "html"),
    ("bailian-models", "阿里云百炼 模型列表（Wan）",
     "https://help.aliyun.com/zh/model-studio/models", "html"),
    ("gemini-pricing", "Gemini API 价格",
     "https://ai.google.dev/gemini-api/docs/pricing", "html"),
    ("gemini-changelog", "Gemini API changelog",
     "https://ai.google.dev/gemini-api/docs/changelog", "html"),
    ("anthropic-news", "Anthropic news",
     "https://www.anthropic.com/news", "html"),
    ("anthropic-pricing", "Anthropic 价格",
     "https://www.anthropic.com/pricing", "html"),
    # ── 竞对与分发 ────────────────────────────────────────────────────────
    ("fal-pricing", "fal 价格（含 Kling 等第三方模型价）",
     "https://fal.ai/pricing", "html"),
    ("replicate-pricing", "Replicate 价格",
     "https://replicate.com/pricing", "html"),
    ("wavespeed-pricing", "WaveSpeedAI 价格",
     "https://wavespeed.ai/pricing", "html"),
    ("higgsfield-pricing", "Higgsfield 定价套餐",
     "https://higgsfield.ai/pricing", "html"),
    ("openrouter-models", "OpenRouter 主力模型价格",
     "https://openrouter.ai/api/v1/models", "openrouter"),
    # ── 市场 ──────────────────────────────────────────────────────────────
    ("aa-video-arena", "Artificial Analysis 视频模型榜",
     ["https://artificialanalysis.ai/video/arena",
      "https://artificialanalysis.ai/text-to-video/arena"], "html"),
]

# OpenRouter: only track these vendors' models, plus anything image/video.
OPENROUTER_PREFIXES = ("openai/", "anthropic/", "google/", "x-ai/", "deepseek/",
                       "moonshotai/", "qwen/", "minimax/")

MIN_TEXT_CHARS = 300      # below this the page is probably JS-rendered
MAX_REPORT_LINES = 15     # per side (added / removed) in one item
LAYOUT_CHANGE_RATIO = 0.5 # >50% of lines changed (and >30 lines) = redesign, re-baseline

_NOISE_RE = re.compile(
    r"(last updated|©|copyright|cookie|all rights reserved|"
    r"\b(19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|\d{1,2}:\d{2}:\d{2})",
    re.IGNORECASE,
)


def _page_text(html_text):
    """Visible text lines of a page, minus scripts/nav/footer and date noise."""
    t = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav)\b.*?</\1>", " ", html_text)
    t = re.sub(r"(?s)<!--.*?-->", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d|td|th|section|article)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    import html as _html
    t = _html.unescape(t)
    lines, seen = [], set()
    for raw in t.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < 2 or _NOISE_RE.search(line) or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _openrouter_lines(json_text):
    lines = []
    for m in json.loads(json_text).get("data", []):
        mid = m.get("id", "")
        if not (mid.startswith(OPENROUTER_PREFIXES) or "image" in mid or "video" in mid):
            continue
        p = m.get("pricing") or {}
        lines.append(f"{mid} | prompt {p.get('prompt')} | completion {p.get('completion')}"
                     f" | image {p.get('image')}")
    return sorted(lines)


def _fetch_lines(urls, kind):
    """Returns (url_used, lines) or raises with the last error."""
    last_err = None
    for url in urls:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=25)
            resp.raise_for_status()
            lines = _openrouter_lines(resp.text) if kind == "openrouter" else _page_text(resp.text)
            return url, lines
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("no url")


def watch_pages():
    """Returns (news_items, health_rows)."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    items, health = [], []

    for slug, name, url, kind in WATCH_PAGES:
        urls = url if isinstance(url, (list, tuple)) else [url]
        snap = SNAPSHOT_DIR / f"{slug}.txt"
        try:
            used_url, lines = _fetch_lines(urls, kind)
        except Exception as e:
            health.append({"source": name, "type": "page", "status": "fail",
                           "note": str(e)[:120]})
            continue

        if kind == "html" and sum(len(l) for l in lines) < MIN_TEXT_CHARS:
            health.append({"source": name, "type": "page", "status": "fail",
                           "note": "正文过短，疑似 JS 渲染，抓不到内容"})
            continue

        if not snap.exists():
            snap.write_text("\n".join(lines), encoding="utf-8")
            health.append({"source": name, "type": "page", "status": "ok",
                           "note": "首次运行，已建立基线"})
            continue

        old = snap.read_text(encoding="utf-8").splitlines()
        old_set, new_set = set(old), set(lines)
        added = [l for l in lines if l not in old_set]
        removed = [l for l in old if l not in new_set]
        changed = len(added) + len(removed)

        if changed == 0:
            health.append({"source": name, "type": "page", "status": "ok", "note": "无变化"})
            continue

        snap.write_text("\n".join(lines), encoding="utf-8")
        if changed > 30 and changed > LAYOUT_CHANGE_RATIO * max(len(old), 1):
            health.append({"source": name, "type": "page", "status": "ok",
                           "note": f"大面积改版（{changed} 行），已重建基线，不报"})
            continue

        def clip(ls):
            return [l[:160] for l in ls[:MAX_REPORT_LINES]]

        summary = "页面相对上次快照的变化。\n新增：\n" + "\n".join(f"+ {l}" for l in clip(added))
        summary += "\n删除：\n" + "\n".join(f"- {l}" for l in clip(removed))
        item = make_item(
            category="news",
            title=f"[页面变动] {name}",
            url=used_url,
            summary="",
            source="页面监测",
            metadata={"watch": True, "slug": slug},
        )
        item["summary"] = summary[:2500]   # keep the diff lines; make_item clips to 500
        item["focus"] = True
        item["importance_hint"] = 5
        items.append(item)
        health.append({"source": name, "type": "page", "status": "ok",
                       "note": f"有变化：+{len(added)} / -{len(removed)} 行"})

    return items, health

"""
Voice broadcast: turn a daily digest into a natural spoken script (DeepSeek)
and synthesize it to an MP3 with edge-tts (free Microsoft neural voices).

Designed for commute listening: the MP3 is hosted under docs/audio/ and also
surfaced via a podcast RSS feed, so listeners can subscribe in any podcast app.
"""

import asyncio
import os
import re

import edge_tts

from generate_site import parse_digest, WEEKDAYS


# News-anchor style zh voice; override with AUDIO_VOICE if desired.
DEFAULT_VOICE = "zh-CN-YunyangNeural"

PODCAST_NAME = "Andy 的 AI 早报"

# 中文男声新闻播音腔约每分钟 270 字：3000-4000 字 ≈ 11-15 分钟。
MIN_SCRIPT_CHARS = 2800

NARRATION_SYSTEM = f"""你是播客《{PODCAST_NAME}》的男主播，要把书面简报改写成口播稿，供听众在通勤路上收听。
听众是 AI 行业从业者，关心 AI 视频生成、图像生成、模型 API 与定价、AI Agent、AI 创作工具。

结构：
1. 开场：问候，报出日期，用三四句话预告今天最重要的三件事。
2. 重点深讲：挑 4-6 条最重要的（优先关注赛道），每条讲清楚三层：发生了什么、为什么重要、对从业者意味着什么（结合简报里的"应用价值"）。
3. 快讯：其余条目按主题合并，快速带过，每条一两句。
4. 今日观察：展开讲趋势判断。
5. 商业机会和研究建议：用聊天的语气讲出来，说清楚依据是哪条新闻。
6. 收尾：一句话总结，然后道别，"我们明天见"。

要求：
- 纯口语，不要 Markdown、星号、链接、"来源"、"重要性五星"这类书面元素。
- 用过渡词串联，不要逐条机械念。
- 专有名词保留英文；数字、英文缩写写成适合朗读的形式。
- 总长度 3000-4000 字（约 11-15 分钟）。宁可在重点条目上多讲背景和影响，也不要不够长。
- 只输出播报稿正文，不要标题、小节名或解释。"""


def _deepseek_script(zh_markdown: str, date_str: str) -> str | None:
    """Ask DeepSeek to rewrite the digest into a spoken script. None on failure."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("[audio] API_KEY not set; using deterministic script.")
        return None

    import requests

    base_url = os.environ.get("API_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("API_MODEL", "deepseek-chat")
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": NARRATION_SYSTEM},
                    {"role": "user", "content":
                        f"今天是 {date_str}。请把下面的简报改写成播报稿：\n\n{zh_markdown}"},
                ],
                "temperature": 0.6,
                "max_tokens": 8000,
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text and len(text) < MIN_SCRIPT_CHARS:
            print(f"[audio] Script only {len(text)} chars; asking for a longer version...")
            resp2 = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": NARRATION_SYSTEM},
                        {"role": "user", "content":
                            f"今天是 {date_str}。请把下面的简报改写成播报稿：\n\n{zh_markdown}"},
                        {"role": "assistant", "content": text},
                        {"role": "user", "content":
                            f"这版只有 {len(text)} 字，太短了。请重写完整的一版，扩到 3000-4000 字："
                            "重点条目多讲背景、影响和对从业者的意义，商业机会和研究建议讲具体。只输出新稿。"},
                    ],
                    "temperature": 0.6,
                    "max_tokens": 8000,
                },
                timeout=240,
            )
            resp2.raise_for_status()
            longer = resp2.json()["choices"][0]["message"]["content"].strip()
            if len(longer) > len(text):
                text = longer
        print(f"[audio] Narration script: {len(text)} chars")
        return text or None
    except Exception as e:  # network / API hiccup → fall back, never block audio
        print(f"[audio] DeepSeek narration failed ({e}); using deterministic script.")
        return None


def _weekday(date_str: str) -> str:
    from datetime import datetime
    try:
        return WEEKDAYS[datetime.strptime(date_str, "%Y-%m-%d").weekday()]
    except Exception:
        return ""


def _fallback_script(zh_markdown: str, date_str: str) -> str:
    """Deterministic spoken script built from the parsed digest (no LLM)."""
    digest = parse_digest(zh_markdown)
    parts = [f"{PODCAST_NAME}，{_weekday(date_str)}，{date_str}。以下是今天的重点。"]
    for cat in digest["categories"]:
        items = cat["items"]
        if not items:
            continue
        parts.append(f"{cat['name']}。")
        for it in items:
            line = it["title"]
            if it["desc"]:
                line += "。" + it["desc"]
            parts.append(line.rstrip("。") + "。")
    if digest["observation"]:
        parts.append("今日观察。" + digest["observation"])
    for name, lines in digest.get("insights", []):
        parts.append(name + "。" + "。".join(l.rstrip("。") for l in lines) + "。")
    parts.append(f"以上就是今天的{PODCAST_NAME}，我们明天见。")
    return "\n".join(parts)


def _clean_for_tts(text: str) -> str:
    """Strip any stray markdown/symbols that would be read awkwardly."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [name](url) -> name
    text = re.sub(r"[*#`>_]+", "", text)                    # md emphasis/headers
    text = re.sub(r"[★☆]+", "", text)                       # stray stars
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


async def _synthesize(text: str, out_path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice).save(out_path)


def generate_audio(zh_markdown: str, date_str: str, audio_dir):
    """Generate {date}.mp3 (+ {date}.txt show-notes script) in audio_dir.

    Returns (mp3_path, script_text) or (None, None) if synthesis failed.
    """
    from pathlib import Path
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    script = _deepseek_script(zh_markdown, date_str) or _fallback_script(zh_markdown, date_str)
    script = _clean_for_tts(script)
    if not script:
        print("[audio] Empty script; skipping audio.")
        return None, None

    voice = os.environ.get("AUDIO_VOICE", DEFAULT_VOICE)
    mp3_path = audio_dir / f"{date_str}.mp3"
    print(f"[audio] Synthesizing {mp3_path.name} with {voice} ({len(script)} chars)...")
    try:
        asyncio.run(_synthesize(script, str(mp3_path), voice))
    except Exception as e:
        print(f"[audio] edge-tts synthesis failed: {e}")
        return None, None

    # Save the script as show notes / transcript.
    (audio_dir / f"{date_str}.txt").write_text(script, encoding="utf-8")
    size = mp3_path.stat().st_size
    print(f"[audio] Saved {mp3_path.name} ({size // 1024} KB)")
    return mp3_path, script


def prune_old_audio(audio_dir, keep: int = 15) -> None:
    """Keep only the `keep` most recent episodes; delete older mp3 + txt.

    Bounds repo size (~3 MB/day). Date-named files sort chronologically, so the
    tail is the newest. Older day pages just lose their player — text content is
    untouched (generate_site only shows a player when the mp3 still exists)."""
    from pathlib import Path
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return
    mp3s = sorted(audio_dir.glob("*.mp3"))
    for mp3 in mp3s[:-keep] if len(mp3s) > keep else []:
        mp3.unlink(missing_ok=True)
        mp3.with_suffix(".txt").unlink(missing_ok=True)
        print(f"[audio] pruned old episode {mp3.stem}")

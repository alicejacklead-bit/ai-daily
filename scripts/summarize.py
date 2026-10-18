"""
AI summarization - supports DeepSeek / OpenAI / any OpenAI-compatible API.

Two passes: one call writes the Chinese digest, then the English version is
translated from it section by section. Asking for both languages in a single
call overflowed the model's 8192-token output cap once digests grew past ~30
items: the English half got cut off mid-item (or never started).
"""

import os
import re
import requests


SYSTEM_PROMPT = """你是一位 AI 行业的资深编辑，为一位 AI 模型 API 平台的海外增长负责人输出每日简报。
他关心：AI 视频生成、图像生成、模型 API 与定价、AI Agent、AI 创作工具、多模态模型。
他读简报是为了两件事：知道行业发生了什么，以及判断这些变化能不能转化为产品、内容或商业机会。

【筛选标准】优先选择，其余直接丢弃：
- 标注 [关注赛道] 的条目优先，同等条件下优先入选
- 重大模型/产品发布、API 上线或调价
- 技术突破（新架构、推理加速、重要 benchmark）
- 有实际使用价值的开源项目
- 产业动向（大厂战略、融资/并购、政策监管）
- 前沿研究（HuggingFace 精选论文）
- 业内热议 / 大佬观点（smol.ai AINews、TLDR AI、Reddit、Hacker News 等聚合源）

【丢弃标准】
- 无实质内容的观点、预测类文章
- 标题党、无具体细节
- 重复或高度相似的报道（只保留最详细的一条，把其他来源一并列在"来源"里）
- 与【近几天已报道】列表中同一事件重复的内容；除非今天有实质性新进展，此时标题写明新进展

【务必保留】重大模型 / 研究发布即使只有单一来源，也必须纳入。

【来源核实（硬性规定）】
- "来源"里的每个 URL 必须从原始资讯里逐字复制，严禁编造、补全或改写链接
- 一件事只要原始资讯里能找到 2 个及以上独立来源，就全部列出，方便交叉核实
- 原始资讯没有给出的数字、日期、价格，不要写

【输出要求】只输出中文版，格式严格遵守：
1. 总计 12-18 条，按类别分组：## 📰 行业新闻 / ## 📄 重要论文 / ## 🔧 开源项目 / ## 🐦 AI 圈热议
2. 每条格式固定：
- **[标题]**：一句话说清楚发生了什么。
  - 重要性：★★★★☆ / 5
  - 核心价值：对开发者/研究者/产品的具体影响（不超过 30 字）
  - 应用价值：能怎么用——落到产品、内容、获客或选型上的一个具体做法（不超过 40 字）
  - 来源：[Source Name](URL), [Source Name 2](URL2)
3. `## 🐦 AI 圈热议` 固定 1-2 条，严禁编造。
4. 条目之后依次输出三个小节，标题一字不改：
### 今日观察
3-4 句话，点出今天最值得关注的趋势或信号，有观点，不废话。
### 商业机会
2-3 条，每条以 "- " 开头：基于今天的事实，指出一个可能的商业机会（谁需要什么、为什么是现在），并写明依据的是哪条新闻。
### 研究建议
2-3 条，每条以 "- " 开头：今天值得深挖或亲手试一下的东西，写清楚具体做什么。
5. 严格基于原文，专有名词保留英文，其余用中文。机会和建议是推断，要写得具体，但不要夸大确定性。
"""


TRANSLATE_PROMPT = """You translate one section of a Chinese AI news digest into English.

Rules:
1. Faithful, item-for-item translation: same number of items, same order, same star ratings, same URLs (copy every URL exactly).
2. Headings keep their emoji: ## 📰 Industry News / ## 📄 Papers / ## 🔧 Open Source / ## 🐦 AI Discourse; "### 今日观察" becomes "### Today's Take", "### 商业机会" becomes "### Opportunities", "### 研究建议" becomes "### Worth Exploring".
3. Each item uses English field labels, exactly in this shape:
- **[Title]**: one sentence on what happened.
  - Importance: ★★★★☆ / 5
  - Why it matters: concrete impact (under ~15 words)
  - How to use it: one concrete application (under ~20 words)
  - Source: [Source Name](URL)
4. Keep proper nouns as they are. Output only the translated Markdown, nothing before or after it.
"""


def _chat(system, user, max_tokens=8192, temperature=0.7):
    """One chat call. Returns (text, finish_reason)."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is required")
    base_url = os.environ.get("API_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("API_MODEL", "deepseek-chat")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason")


def _trim_partial(text, finish_reason, label):
    """If the output hit the token cap, drop the item that was cut off mid-way."""
    if finish_reason != "length":
        return text.strip()
    cut = max(text.rfind("\n- **"), text.rfind("\n### "))
    print(f"[summarize] WARNING: {label} hit the output cap; dropping the unfinished last item.")
    return (text[:cut] if cut > 0 else text).strip()


def split_sections(markdown):
    """Split a digest body at its '## ' headings ('### 今日观察' stays with the last one)."""
    return [s.strip() for s in re.split(r"(?m)^(?=## )", markdown) if s.strip()]


def translate_digest(zh_text):
    """Translate a Chinese digest body to English, one section per call.

    A section that fails to translate is left out rather than failing the whole
    run; returns None only when nothing could be translated."""
    parts = []
    for section in split_sections(zh_text):
        try:
            text, reason = _chat(TRANSLATE_PROMPT, section, temperature=0.3)
        except Exception as e:  # network / API error on one section
            print(f"[summarize] WARNING: translating a section failed ({e}); skipping it.")
            continue
        parts.append(_trim_partial(text, reason, "English translation"))
    return "\n\n".join(parts) if parts else None


def format_raw_content(data):
    """Format fetched data into a text block for the AI."""
    sections = []

    if data.get("news"):
        sections.append("=== 新闻资讯 ===")
        for item in data["news"]:
            sections.append(
                f"{'[关注赛道] ' if item.get('focus') else ''}[{item['source']}] {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Published: {item.get('published') or 'unknown'}\n"
                f"Importance hint: {item.get('importance_hint', 3)}/5\n"
                f"{item['summary']}\n"
            )

    if data.get("papers"):
        sections.append("=== 学术论文 ===")
        for item in data["papers"]:
            sections.append(
                f"{'[关注赛道] ' if item.get('focus') else ''}[{item['source']}] {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Published: {item.get('published') or 'unknown'}\n"
                f"Importance hint: {item.get('importance_hint', 3)}/5\n"
                f"{item['summary']}\n"
            )

    if data.get("projects"):
        sections.append("=== 开源项目 ===")
        for item in data["projects"]:
            sections.append(
                f"{'[关注赛道] ' if item.get('focus') else ''}[{item['source']}] {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Importance hint: {item.get('importance_hint', 3)}/5\n"
                f"{item['summary']}\n"
            )

    return "\n".join(sections)


def _norm(url):
    from sources import normalize_url
    return normalize_url(url)


def verify_sources(markdown, data):
    """Drop source links the model did not take from today's raw data.

    An item left with no verified source is removed entirely; returns
    (clean_markdown, report dict)."""
    allowed = {
        _norm(item.get("url", ""))
        for items in data.values() for item in items if item.get("url")
    }
    link_re = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
    out, item_buf = [], []
    report = {"links_dropped": [], "items_dropped": []}

    def flush():
        if not item_buf:
            return
        src_lines = [l for l in item_buf if re.match(r"^\s*-\s+(来源|Sources?)[：:]", l)]
        has_src = any(link_re.search(l) for l in src_lines)
        title = item_buf[0].strip()[:80]
        if src_lines and not has_src:
            report["items_dropped"].append(title)
        else:
            out.extend(item_buf)
        item_buf.clear()

    for line in markdown.splitlines():
        if re.match(r"^\s*-\s+(来源|Sources?)[：:]", line):
            kept = []
            for name, url in link_re.findall(line):
                if _norm(url) in allowed:
                    kept.append(f"[{name}]({url})")
                else:
                    report["links_dropped"].append(url)
            label = re.match(r"^(\s*-\s+(?:来源|Sources?)[：:]\s*)", line).group(1)
            line = label + ", ".join(kept) if kept else label.rstrip()
            item_buf.append(line)
            continue
        if re.match(r"^-\s+\*\*", line):          # new item
            flush()
            item_buf.append(line)
        elif item_buf and re.match(r"^\s+-\s+", line):  # sub-field of current item
            item_buf.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out), report


def _wrap(text, date_str, model, title):
    return f"""# {title} - {date_str}

{text}

---
*Generated by Andy 的 AI 早报 using {model}*
"""


def summarize(data, date_str, recent_titles=None):
    """Generate daily digest via OpenAI-compatible API (DeepSeek by default).

    Returns (zh_markdown, en_markdown). en_markdown is None if the English
    translation failed entirely.
    """
    model = os.environ.get("API_MODEL", "deepseek-chat")

    raw_content = format_raw_content(data)
    if not raw_content.strip():
        return (
            f"# AI Daily Digest - {date_str}\n\n> No content fetched today.\n",
            None,
        )

    user_prompt = f"""今天是 {date_str}。

以下是今天收集到的 AI 领域原始资讯，请整理成每日简报。
请优先选择重要性高、来源可靠、对 AI builder 有实际参考价值的内容：

{raw_content}"""
    if recent_titles:
        user_prompt += "\n\n=== 近几天已报道（同一事件不要重复，除非有实质新进展）===\n" + "\n".join(
            f"- {t}" for t in recent_titles)

    print(f"Calling {model} for the Chinese digest...")
    zh_text, reason = _chat(SYSTEM_PROMPT, user_prompt)
    zh_text = _trim_partial(zh_text, reason, "Chinese digest")
    zh_text, report = verify_sources(zh_text, data)
    print(f"[verify] dropped {len(report['links_dropped'])} unverified links, "
          f"{len(report['items_dropped'])} items with no verified source")
    for u in report["links_dropped"]:
        print(f"  - link not in raw data: {u}")
    for t in report["items_dropped"]:
        print(f"  - item removed: {t}")

    print("Translating to English, one section at a time...")
    en_text = translate_digest(zh_text)
    if not en_text:
        print("[summarize] WARNING: English translation failed; zh only.")

    zh_md = _wrap(zh_text, date_str, model, "Andy 的 AI 早报")
    en_md = _wrap(en_text, date_str, model, "Andy's AI Daily") if en_text else None
    return zh_md, en_md

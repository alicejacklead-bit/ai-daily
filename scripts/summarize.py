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


SYSTEM_PROMPT = """你是一位资深 AI 行业分析师，为一位 AI 模型 API 平台（以视频生成模型为主）的海外 GTM 负责人写每日简报。
他读简报有四个目的：
1. 看懂市场局势：头部厂商和模型竞争、价格走势、资本流向、政策变化；
2. 跟踪自己的赛道：视频/图像生成模型、模型 API 定价与分发渠道、AI 视频创作平台；
3. 积累 Agent 认知：这是他下一阶段要进入的方向；
4. 有谈资：和海外 AI 公司创始人、CTO 聊天时能接住话题。
他最大的不满是：真正关心的内容被一两句话带过。所以宁可少选，选中的要讲透。

【分区定义与入选标准】
- 市场局势（2-3 条）：头部厂商与模型竞争格局、行业价格走势、重大融资并购、政策监管。必须讲出"这说明什么趋势"。
- 赛道深挖（2-3 条）：只选以下内容——视频/图像生成模型的新版本、能力或价格变化；模型 API 定价、聚合平台、分发渠道的变动；AI 视频创作平台的融资、上线、换模型；C 端创作工具的定价和套餐变化。讲透具体数字和影响。
- Agent 前沿（1-2 条）：Agent 框架、产品、方法上的重要进展。讲清是什么、解决什么问题、为什么值得关注。
- 快讯（3-5 条）：其他值得知道的事，一句话。不够就少写，严禁凑数。论文和开源项目只有真正重要时才进快讯或 Agent 前沿。
- 标注 [关注赛道] 的条目同等条件下优先。
- 标题为 [页面变动] 的条目来自官方价格页 / changelog 的快照对比，摘要里列出了新增（+）和删除（-）的行：
  · 若涉及价格、套餐、新模型上线，优先放入赛道深挖或市场局势，写明变化前后的具体数字（只用摘要里出现的数字）；
  · 若只是文案、导航、无关内容的变化，直接丢弃；
  · 只有能力更新、没有价格变化时，只写能力，不要虚构价格。

【丢弃标准】
- 无实质内容的观点、预测类文章；标题党、无具体细节
- 重复或高度相似的报道（只保留最详细的一条，把其他来源一并列在"来源"里）
- 与【近几天已报道】列表中同一事件重复的内容；除非今天有实质性新进展，此时标题写明新进展

【来源核实（硬性规定）】
- "来源"里的每个 URL 必须从原始资讯里逐字复制，严禁编造、补全或改写链接
- 一件事只要原始资讯里能找到 2 个及以上独立来源，就全部列出，方便交叉核实
- 原始资讯没有给出的数字、日期、价格，不要写

【谈资角度规则】只写通用切入点（一个可以抛给对方的问题或观点），不得提及任何具体客户、合作方或"我们的业务"。

【输出格式】只输出中文版，严格按以下顺序和格式：

第一行：
> 🎯 今日一句话：当天最重要的一个判断，一句话，有观点。

然后依次四个分区，标题一字不改：
## 🌐 市场局势
## 🎬 赛道深挖
## 🤖 Agent 前沿
## ⚡ 快讯

市场局势、赛道深挖、Agent 前沿的每条格式：
- **[标题]**：发生了什么，2-3 句，写清主体和具体数字（价格、参数、金额、日期）。
  - 重要性：★★★★☆ / 5
  - 为什么重要：趋势判断与影响，40-70 字。赛道深挖要写对 API 平台和视频生成赛道的影响；Agent 前沿要写它解决什么问题。
  - 谈资角度：和海外创始人 / CTO 聊这件事时的一个切入点，不超过 40 字。
  - 来源：[Source Name](URL), [Source Name 2](URL2)

快讯每条格式：
- **[标题]**：一句话说清发生了什么。
  - 来源：[Source Name](URL)

四个分区之后，依次输出两个小节，标题一字不改：
### 今日观察
3-4 句话，把今天的信号串成趋势判断，有观点，不废话。
### 商业机会
2 条，每条以 "- " 开头：基于今天的事实，指出一个可能的商业机会（谁需要什么、为什么是现在），写明依据哪条新闻。

专有名词保留英文，其余用中文。判断和机会是推断，要具体，但不要夸大确定性。
"""


TRANSLATE_PROMPT = """You translate one section of a Chinese AI news digest into English.

Rules:
1. Faithful, item-for-item translation: same number of items, same order, same star ratings, same URLs (copy every URL exactly).
2. Headings keep their emoji and map exactly: "## 🌐 市场局势" -> "## 🌐 Market Landscape"; "## 🎬 赛道深挖" -> "## 🎬 Track Deep Dive"; "## 🤖 Agent 前沿" -> "## 🤖 Agent Frontier"; "## ⚡ 快讯" -> "## ⚡ Quick Hits"; "### 今日观察" -> "### Today's Take"; "### 商业机会" -> "### Opportunities".
3. The opening line "> 🎯 今日一句话：..." becomes "> 🎯 One-liner: ...".
4. Item field labels map exactly: 重要性 -> Importance; 为什么重要 -> Why it matters; 谈资角度 -> Talking point; 来源 -> Source. Keep the same indentation and bullet shape:
- **[Title]**: what happened.
  - Importance: ★★★★☆ / 5
  - Why it matters: ...
  - Talking point: ...
  - Source: [Source Name](URL)
5. Keep proper nouns as they are. Output only the translated Markdown, nothing before or after it.
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
按系统提示里的四个分区筛选，宁少勿凑：

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

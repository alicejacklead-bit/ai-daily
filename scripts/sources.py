"""
Data source fetchers - all free, no AI tokens needed.
Each function returns a list of dicts with stable fields for traceability.
"""

import html
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import requests


AI_KEYWORDS = [
    "ai", "ml", "llm", "gpt", "transformer", "neural", "deep-learning",
    "machine-learning", "nlp", "diffusion", "agent", "rag", "embedding",
    "model", "inference", "fine-tun", "lora", "vision", "multimodal",
    "chatbot", "langchain", "openai", "anthropic", "gemini", "claude",
]

HIGH_SIGNAL_KEYWORDS = [
    "release", "launch", "open source", "benchmark", "state-of-the-art",
    "sota", "reasoning", "inference", "agent", "multimodal", "model",
    "api", "eval", "safety", "chip", "funding", "acquisition",
    "发布", "开源", "融资", "收购", "基准", "推理", "模型", "智能体",
    "多模态", "安全", "芯片",
]

# 关注赛道：命中的条目会提高重要性提示，并在送给模型时标注 [关注赛道]。
# 只写赛道、不写具体公司名（仓库是公开的）。
FOCUS_KEYWORDS = [
    # AI 视频生成
    "video generation", "text-to-video", "image-to-video", "video model", "ai video",
    "视频生成", "文生视频", "图生视频", "视频模型", "ai视频", "ai 视频",
    # 图像生成
    "image generation", "text-to-image", "image model", "图像生成", "文生图", "生图",
    # 模型 API 与定价
    "api pricing", "price cut", "pricing", "per token", "rate limit", "inference cost",
    "api", "定价", "降价", "价格", "调用", "推理成本",
    # AI Agent
    "agent", "agentic", "智能体",
    # AI 创作工具
    "creative tool", "creator", "editing", "storyboard", "创作", "剪辑", "分镜",
    # 多模态模型
    "multimodal", "omni", "多模态",
]

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"}


def clean_text(value, max_chars=500):
    """Strip HTML and whitespace noise from source summaries."""
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def normalize_url(url):
    """Normalize URLs so the same story is easier to dedupe."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        urlencode(query, doseq=True),
        "",
    ))


def normalize_title(title):
    return re.sub(r"\W+", "", (title or "").lower())


_FOCUS_RE = re.compile(
    "|".join(
        (r"\b" + re.escape(k) + r"\b") if k.isascii() else re.escape(k)
        for k in FOCUS_KEYWORDS
    ),
    re.IGNORECASE,
)


def is_focus(title, summary=""):
    """命中关注赛道。英文按词边界匹配，避免 api 命中 capital 这类误判。"""
    return bool(_FOCUS_RE.search(f"{title} {summary}"))


def estimate_importance(category, title, summary, source):
    """Give the model a transparent first-pass signal without replacing editorial judgment."""
    text = f"{title} {summary} {source}".lower()
    score = 3

    if category in {"papers", "projects"}:
        score += 1
    if any(keyword in text for keyword in HIGH_SIGNAL_KEYWORDS):
        score += 1
    if any(name in text for name in ["openai", "anthropic", "google", "deepmind", "meta", "microsoft", "nvidia"]):
        score += 1
    if is_focus(title, summary):
        score += 1

    return max(1, min(5, score))


def make_item(category, title, url, summary, source, published=None, metadata=None):
    cleaned_summary = clean_text(summary)
    return {
        "category": category,
        "title": clean_text(title, max_chars=220),
        "url": url or "",
        "normalized_url": normalize_url(url),
        "summary": cleaned_summary,
        "source": source,
        "published": published,
        "importance_hint": estimate_importance(category, title, cleaned_summary, source),
        "focus": is_focus(title, cleaned_summary),
        "metadata": metadata or {},
    }


def dedupe_items(items):
    """Deduplicate by normalized URL first, then normalized title."""
    seen = {}
    url_index = {}
    title_index = {}

    for item in items:
        url_key = item.get("normalized_url")
        title_key = normalize_title(item.get("title"))
        key = url_index.get(url_key) or title_index.get(title_key)

        if not key:
            key = url_key or title_key
        if not key:
            continue

        previous = seen.get(key)
        if not previous:
            seen[key] = item
            if url_key:
                url_index[url_key] = key
            if title_key:
                title_index[title_key] = key
            continue

        if len(item.get("summary", "")) > len(previous.get("summary", "")):
            item["source"] = f"{previous['source']}, {item['source']}"
            item["importance_hint"] = max(previous["importance_hint"], item["importance_hint"])
            seen[key] = item
            if url_key:
                url_index[url_key] = key
            if title_key:
                title_index[title_key] = key
        else:
            previous["source"] = f"{previous['source']}, {item['source']}"
            previous["importance_hint"] = max(previous["importance_hint"], item["importance_hint"])

    return list(seen.values())


def fetch_huggingface_papers():
    """Fetch today's papers from HuggingFace Daily Papers API."""
    items = []
    try:
        resp = requests.get("https://huggingface.co/api/daily_papers", timeout=30)
        resp.raise_for_status()
        papers = resp.json()
        for p in papers[:30]:  # top 30
            paper = p.get("paper", {})
            paper_id = paper.get("id", "")
            items.append(make_item(
                category="papers",
                title=paper.get("title", ""),
                url=f"https://huggingface.co/papers/{paper_id}",
                summary=paper.get("summary", ""),
                source="HuggingFace Papers",
                published=paper.get("publishedAt") or paper.get("submittedOn"),
                metadata={"paper_id": paper_id},
            ))
    except Exception as e:
        print(f"[HuggingFace Papers] Error: {e}")
    return items


def fetch_github_trending():
    """Fetch trending repos from OSSInsight API (free, no auth)."""
    items = []
    try:
        resp = requests.get(
            "https://api.ossinsight.io/v1/trends/repos?period=past_24_hours",
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json().get("data", {}).get("rows", [])
        # Filter for AI/ML related repos by keywords
        for repo in rows:
            desc = (repo.get("description") or "").lower()
            name = (repo.get("repo_name") or "").lower()
            lang = (repo.get("primary_language") or "").lower()
            text = f"{desc} {name} {lang}"
            if any(kw in text for kw in AI_KEYWORDS):
                repo_name = repo.get("repo_name", "")
                items.append(make_item(
                    category="projects",
                    title=f"{repo_name} ⭐{repo.get('stars', 0)}",
                    url=f"https://github.com/{repo_name}",
                    summary=repo.get("description", "") or "No description",
                    source="GitHub Trending",
                    metadata={
                        "stars": repo.get("stars", 0),
                        "language": repo.get("primary_language"),
                    },
                ))
        items = items[:15]  # cap at 15
    except Exception as e:
        print(f"[GitHub Trending] Error: {e}")
    return items


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
}


def fetch_github_trending_page(period="daily"):
    """Scrape github.com/trending for genuinely hot repos — higher quality and
    more recognizable than the OSSInsight 24h-star list. Keeps AI-related ones."""
    items = []
    try:
        resp = requests.get(
            f"https://github.com/trending?since={period}",
            headers=BROWSER_HEADERS, timeout=20,
        )
        resp.raise_for_status()
        rows = re.findall(r'<article class="Box-row">(.*?)</article>', resp.text, re.S)
        for row in rows:
            h2 = re.search(r'<h2.*?</h2>', row, re.S)
            if not h2:
                continue
            m = re.search(r'href="/([^"/]+/[^"/]+)"', h2.group(0))
            if not m:
                continue
            repo = m.group(1)
            dm = re.search(r'<p class="col-9[^"]*"[^>]*>(.*?)</p>', row, re.S)
            desc = clean_text(dm.group(1)) if dm else ""
            sm = re.search(r'([\d,]+)\s+stars today', row)
            today = sm.group(1) if sm else ""
            blob = f"{repo} {desc}".lower()
            if not any(kw in blob for kw in AI_KEYWORDS):
                continue
            note = f"（GitHub Trending · 今日 +{today} star）" if today else "（GitHub Trending）"
            items.append(make_item(
                category="projects",
                title=f"{repo} ⭐+{today}/天" if today else repo,
                url=f"https://github.com/{repo}",
                summary=f"{desc} {note}".strip(),
                source="GitHub Trending",
                metadata={"stars_today": today},
            ))
    except Exception as e:
        print(f"  [GitHub Trending page] Error: {e}")
    return items


def fetch_rss_feed(feed_url, source_name, max_items=10, cutoff_hours=24):
    """Generic RSS feed fetcher."""
    items = []
    try:
        resp = requests.get(feed_url, headers=BROWSER_HEADERS, timeout=20)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=cutoff_hours)
        # Scan a generous window so high-volume feeds (HF/OpenAI blogs with
        # hundreds of entries) still get correctly date-filtered.
        for entry in feed.entries[:max(max_items * 3, 40)]:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                entry_date = datetime(*published[:6], tzinfo=timezone.utc)
                if entry_date < cutoff:
                    continue
                published_at = entry_date.isoformat()
            else:
                # 没有发布时间就无法核实是否在时间窗内，宁缺毋滥
                continue
            items.append(make_item(
                category="news",
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                summary=entry.get("summary", ""),
                source=source_name,
                published=published_at,
                metadata={"feed_url": feed_url},
            ))
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"  [{source_name}] Error: {e}")
    return items


def fetch_arxiv(max_items=12, cutoff_days=2):
    """Recent cs.AI/CL/CV/LG papers straight from arXiv — a non-curated safety
    net so important papers that don't make HuggingFace's daily top list still
    have a chance to surface."""
    items = []
    try:
        query = (
            "http://export.arxiv.org/api/query?"
            "search_query=cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.AI"
            "&sortBy=submittedDate&sortOrder=descending&max_results=40"
        )
        resp = requests.get(query, headers=BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        cutoff = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        for entry in feed.entries:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            published_at = None
            if published:
                entry_date = datetime(*published[:6], tzinfo=timezone.utc)
                if entry_date < cutoff:
                    continue
                published_at = entry_date.isoformat()
            items.append(make_item(
                category="papers",
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                summary=entry.get("summary", ""),
                source="arXiv",
                published=published_at,
            ))
            if len(items) >= max_items:
                break
    except Exception as e:
        print(f"  [arXiv] Error: {e}")
    return items


def fetch_hacker_news():
    """Fetch top AI stories from Hacker News via Algolia API."""
    items = []
    try:
        from datetime import datetime, timezone, timedelta
        since = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp())
        # Pull recent high-point stories (no text query — a multi-term query is
        # ANDed and returns almost nothing); the AI_KEYWORDS title filter below
        # narrows it. The relevance /search endpoint now 400s, so use by-date.
        url = (
            f"https://hn.algolia.com/api/v1/search_by_date"
            f"?tags=story"
            f"&numericFilters=created_at_i>{since},points>10"
            f"&hitsPerPage=50"
        )
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])

        for hit in hits:
            title = hit.get("title", "")
            # Filter for AI-relevant titles
            if not any(kw in title.lower() for kw in AI_KEYWORDS):
                continue
            items.append(make_item(
                category="news",
                title=title,
                url=hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                summary=f"HN points: {hit.get('points', 0)}, comments: {hit.get('num_comments', 0)}",
                source="Hacker News",
                published=hit.get("created_at"),
            ))
        items = items[:15]
    except Exception as e:
        print(f"  [Hacker News] Error: {e}")
    return items


# Public RSSHub instances, tried in order (the first that returns items wins).
# Public instances are individually flaky, so the fallback list keeps the
# rsshub-backed sources (36Kr, SSPAI) resilient.
RSSHUB_INSTANCES = [
    "https://rsshub.rssforever.com",
    "https://hub.slarker.me",
    "https://rsshub.app",
]


def rsshub(path):
    """Candidate URLs for an rsshub route across all configured instances."""
    return [instance + path for instance in RSSHUB_INSTANCES]


# RSS sources format: (url_or_url_list, name, max_items, cutoff_hours)
# url may be a list of fallback URLs; the first that yields items is used.
# 新闻类严格 24 小时；官方博客 / 周更 Newsletter 放宽到 72 小时，
# 重复由 main.py 的跨天去重兜底。
NEWS_H = 24
SLOW_H = 72
RSS_SOURCES = [
    # ── 中文 AI 媒体 ─────────────────────────────────────────────────────────
    (rsshub("/36kr/search/articles/ai"),                             "36Kr AI",    10, NEWS_H),
    ("https://www.qbitai.com/rss",                                   "量子位",     10, NEWS_H),
    ("https://www.infoq.cn/feed.xml",                                "InfoQ 中文",  8, NEWS_H),
    (rsshub("/sspai/tag/AI"),                                        "SSPAI AI",   8, NEWS_H),

    # ── 英文科技媒体 ─────────────────────────────────────────────────────────
    ("https://venturebeat.com/feed/",                                "VentureBeat",      8, NEWS_H),
    ("https://techcrunch.com/category/artificial-intelligence/feed/","TechCrunch AI",   10, NEWS_H),
    ("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "The Verge AI",10, NEWS_H),
    ("https://www.wired.com/feed/tag/ai/latest/rss",                 "Wired AI",         8, NEWS_H),
    ("https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss", "IEEE Spectrum AI", 8, NEWS_H),
    ("https://feeds.arstechnica.com/arstechnica/technology-lab",     "Ars Technica AI",  8, NEWS_H),

    # ── AI 垂直媒体 / 个人博客 ───────────────────────────────────────────────
    ("https://the-decoder.com/feed/",                                "The Decoder",     12, NEWS_H),
    ("https://www.marktechpost.com/feed/",                           "MarkTechPost",    12, NEWS_H),
    ("https://simonwillison.net/atom/everything/",                   "Simon Willison",   8, NEWS_H),

    # ── 官方实验室博客（低频，放宽到 72 小时）────────────────────────────────
    ("https://huggingface.co/blog/feed.xml",                         "Hugging Face Blog", 6, SLOW_H),
    ("https://deepmind.google/blog/rss.xml",                         "Google DeepMind",   6, SLOW_H),
    ("https://blog.google/technology/ai/rss/",                       "Google AI Blog",    6, SLOW_H),
    ("https://openai.com/news/rss.xml",                              "OpenAI",            6, SLOW_H),

    # ── AI 圈讨论聚合 ────────────────────────────────────────────────────────
    ("https://news.smol.ai/rss.xml",                                 "smol.ai AINews",    5, NEWS_H),
    ("https://tldr.tech/api/rss/ai",                                 "TLDR AI",           6, NEWS_H),

    # ── 社区（best-effort，云端 IP 可能被限流，失败自动跳过）─────────────────
    ("https://www.reddit.com/r/LocalLLaMA/top/.rss?t=day",           "r/LocalLLaMA",      8, NEWS_H),
    ("https://www.reddit.com/r/aivideo/top/.rss?t=day",              "r/aivideo",         6, NEWS_H),

    # ── 周更 Newsletter（放宽到 72 小时）─────────────────────────────────────
    ("https://www.oneusefulthing.org/feed",   "One Useful Thing (Mollick)", 5, SLOW_H),
    ("https://www.lennysnewsletter.com/feed", "Lenny's Newsletter",         5, SLOW_H),
]


def fetch_all():
    """Fetch from all sources, return categorized data."""
    print("Fetching HuggingFace Papers...")
    papers = fetch_huggingface_papers()
    print(f"  Got {len(papers)} papers")

    print("Fetching arXiv (cs.AI/CL/CV/LG)...")
    arxiv_items = fetch_arxiv()
    print(f"  Got {len(arxiv_items)} arXiv papers")
    papers.extend(arxiv_items)

    print("Fetching GitHub Trending (official page)...")
    trending_page = fetch_github_trending_page()
    print(f"  Got {len(trending_page)} trending repos")

    print("Fetching GitHub Trending (OSSInsight)...")
    oss_projects = fetch_github_trending()
    print(f"  Got {len(oss_projects)} projects")

    # Official trending first (higher quality), OSSInsight as backup/supplement
    projects = trending_page + oss_projects

    print("Fetching Hacker News...")
    hn_items = fetch_hacker_news()
    print(f"  Got {len(hn_items)} items")

    print("Fetching RSS feeds...")
    news = list(hn_items)
    for url, name, max_items, cutoff_hours in RSS_SOURCES:
        candidates = url if isinstance(url, (list, tuple)) else [url]
        feed_items = []
        for candidate in candidates:
            feed_items = fetch_rss_feed(candidate, name, max_items=max_items, cutoff_hours=cutoff_hours)
            if feed_items:  # first instance that works wins
                break
        print(f"  [{name}] Got {len(feed_items)} items")
        news.extend(feed_items)

    data = {
        "papers": dedupe_items(papers),
        "projects": dedupe_items(projects),
        "news": dedupe_items(news),
    }

    deduped_total = sum(len(v) for v in data.values())
    raw_total = len(papers) + len(projects) + len(news)
    print(f"Deduped {raw_total} raw items to {deduped_total} unique items")
    return data

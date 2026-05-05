#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Papers Trending Daily Report Generator

Fetches trending papers from HuggingFace, enriches with arXiv metadata,
generates a daily Markdown report, and optionally publishes to WeChat drafts.
"""

import os
import re
import sys
import json
import time
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, quote
from textwrap import shorten

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TRENDING_URL = "https://huggingface.co/papers/trending"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
TOP_N = 5

REPO_ROOT = Path(__file__).parent.parent
DAILY_DIR = REPO_ROOT / "daily"
README_PATH = REPO_ROOT / "README.md"

# Optional LLM API for Chinese summarization
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# WeChat Official Account (MP) credentials
WECHAT_APPID = os.getenv("WECHAT_APPID", "")
WECHAT_APPSECRET = os.getenv("WECHAT_APPSECRET", "")
WECHAT_THUMB_MEDIA_ID = os.getenv("WECHAT_THUMB_MEDIA_ID", "")

# ---------------------------------------------------------------------------
# HF Scraping
# ---------------------------------------------------------------------------

def fetch_hf_trending() -> list[dict]:
    """Fetch and parse HuggingFace trending papers page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    resp = requests.get(HF_TRENDING_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    papers = []
    seen_ids = set()

    # Strategy 1: look for <a> tags linking to /papers/xxxx.xxxxx
    for a in soup.find_all("a", href=re.compile(r"/papers/(\d{4}\.\d{4,5})")):
        href = a.get("href", "")
        match = re.search(r"/papers/(\d{4}\.\d{4,5})", href)
        if not match:
            continue
        arxiv_id = match.group(1)
        if arxiv_id in seen_ids:
            continue

        title = _extract_title(a)
        card = a.find_parent(["article", "div", "li"], class_=re.compile("paper|card|item"))
        if card is None:
            card = a.find_parent(["div", "li"])

        upvotes = _extract_upvotes(card) if card else 0
        authors = _extract_authors(card) if card else ""

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "upvotes": upvotes,
            "authors": authors,
        })
        seen_ids.add(arxiv_id)

        if len(papers) >= TOP_N:
            break

    # Strategy 2: if no papers found via links, try to find JSON data in script tags
    if not papers:
        for script in soup.find_all("script"):
            text = script.string or ""
            if "papers" in text.lower() or "trending" in text.lower():
                try:
                    # Try to extract any JSON that looks like paper data
                    for match in re.finditer(r'"(arxiv_id|id)":"(\d{4}\.\d{4,5})"', text):
                        arxiv_id = match.group(2)
                        if arxiv_id in seen_ids:
                            continue
                        # try to find title/upvotes nearby in the JSON blob
                        title = _extract_json_field(text, match.start(), "title") or arxiv_id
                        upvotes = _extract_json_field(text, match.start(), "upvotes", "num") or 0
                        authors = _extract_json_field(text, match.start(), "authors") or ""
                        papers.append({
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "upvotes": int(upvotes) if upvotes else 0,
                            "authors": authors,
                        })
                        seen_ids.add(arxiv_id)
                        if len(papers) >= TOP_N:
                            break
                except Exception:
                    pass
                if len(papers) >= TOP_N:
                    break

    return papers[:TOP_N]


def _extract_title(tag) -> str:
    """Best-effort title extraction from a tag."""
    if tag.get_text(strip=True):
        return tag.get_text(strip=True)
    # Try aria-label or title attribute
    return tag.get("aria-label", tag.get("title", "Unknown"))


def _extract_upvotes(card) -> int:
    """Best-effort upvote extraction from a paper card."""
    if card is None:
        return 0
    text = card.get_text(separator=" ", strip=True)
    # Patterns like "123", "1.2k", "▲ 45", "↑ 10"
    patterns = [
        r"(\d[\d.]*)\s*[kK]\s*(?:upvotes?|likes?|votes?|\u2191|\u25b2|\U0001F44D)?",
        r"(?:upvotes?|likes?|votes?|\u2191|\u25b2|\U0001F44D)\s*(\d[\d.]*)",
        r"(\d+)\s*(?:\u2191|\u25b2|\U0001F44D)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            if "k" in val.lower():
                return int(float(val.lower().replace("k", "")) * 1000)
            return int(float(val))
    return 0


def _extract_authors(card) -> str:
    """Best-effort author extraction from a paper card."""
    if card is None:
        return ""
    # Look for text patterns like "by Author Name" or author links
    text = card.get_text(separator=" ", strip=True)
    # Try to find author list after title, before upvotes
    by_match = re.search(r"by\s+([^0-9]{2,80}?)(?:\s+\d|\s+\u2191|\s+\u25b2|$)", text, re.IGNORECASE)
    if by_match:
        return by_match.group(1).strip(" ,")
    # Look for any tag that might contain author names
    for sel in ["span", "div", "p"]:
        elems = card.find_all(sel, string=re.compile(r"^[A-Z][a-z]+\s+[A-Z][a-z]+"))
        if elems:
            return elems[0].get_text(strip=True)
    return ""


def _extract_json_field(text: str, pos: int, key: str, kind: str = "str") -> str | int | None:
    """Roughly extract a JSON field value near a position."""
    snippet = text[max(0, pos - 500):pos + 500]
    pat = rf'"{key}":\s*"([^"]{{0,300}})"'
    if kind == "num":
        pat = rf'"{key}":\s*(\d+)'
    m = re.search(pat, snippet)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# arXiv enrichment
# ---------------------------------------------------------------------------

def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """Fetch paper metadata from arXiv API."""
    if not arxiv_ids:
        return {}
    ids = ",".join(arxiv_ids)
    params = {"id_list": ids, "max_results": len(arxiv_ids)}
    resp = requests.get(ARXIV_API_URL, params=params, timeout=60)
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.content)
    entries = root.findall("atom:entry", ns)

    meta = {}
    for entry in entries:
        id_url = entry.findtext("atom:id", "", ns)
        match = re.search(r"(\d{4}\.\d{4,5})", id_url)
        if not match:
            continue
        aid = match.group(1)
        title = entry.findtext("atom:title", "", ns).replace("\n", " ").strip()
        summary = entry.findtext("atom:summary", "", ns).replace("\n", " ").strip()
        authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
        published = entry.findtext("atom:published", "", ns)[:10]

        meta[aid] = {
            "title": title,
            "summary": summary,
            "authors": ", ".join(authors) if authors else "",
            "published": published,
        }
    return meta


# ---------------------------------------------------------------------------
# LLM summarization (optional)
# ---------------------------------------------------------------------------

def summarize_with_llm(text: str) -> str:
    """Generate a structured Chinese summary using an LLM API."""
    if not LLM_API_KEY:
        return ""
    prompt = (
        "请用中文为下面这篇论文摘要撰写一个结构化的简短总结（100-150字），"
        "分为四个要点：问题、方法、技术、结果。使用简洁的学术中文。\n\n"
        f"摘要：{text}\n"
    )
    try:
        resp = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一位AI论文解读专家，擅长用中文精准概括论文要点。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 300,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"LLM summarization failed: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def categorize(title: str, summary: str) -> str:
    """Simple rule-based category tagging."""
    text = (title + " " + summary).lower()
    tags = []
    if any(k in text for k in ["agent", "multi-agent", "tool use", "autonomous"]):
        tags.append("🤖 Agent")
    if any(k in text for k in ["llm", "language model", "transformer", "gpt", "token", "pretrain"]):
        tags.append("💬 LLM")
    if any(k in text for k in ["vision", "image", "video", "diffusion", "generative"]):
        tags.append("🎨 生成模型")
    if any(k in text for k in ["rl", "reinforcement", "policy", "reward"]):
        tags.append("🎯 强化学习")
    if any(k in text for k in ["robot", "embodied", "manipulation", "navigation"]):
        tags.append("🦾 机器人")
    if any(k in text for k in ["multimodal", "audio", "speech", "voice"]):
        tags.append("🔊 多模态")
    if not tags:
        tags.append("📚 综合")
    return " · ".join(tags[:2])


def generate_report(papers: list[dict]) -> str:
    """Generate the daily Markdown report."""
    today = datetime.datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    date_cn = today.strftime("%Y年%m月%d日")

    # Fetch arXiv metadata
    arxiv_meta = fetch_arxiv_metadata([p["arxiv_id"] for p in papers])

    # Merge metadata
    for p in papers:
        meta = arxiv_meta.get(p["arxiv_id"], {})
        p["title"] = meta.get("title") or p["title"]
        p["authors"] = meta.get("authors") or p["authors"]
        p["published"] = meta.get("published") or ""
        p["summary"] = meta.get("summary", "")
        p["category"] = categorize(p["title"], p.get("summary", ""))
        # Optional LLM Chinese summary
        p["cn_summary"] = summarize_with_llm(p["summary"]) if p["summary"] else ""

    total_upvotes = sum(p.get("upvotes", 0) for p in papers)
    max_upvotes = max((p.get("upvotes", 0) for p in papers), default=0)
    avg_upvotes = round(total_upvotes / len(papers), 1) if papers else 0

    # Category counts for trend insight
    cat_counts = {}
    for p in papers:
        for c in p["category"].split(" · "):
            cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_insight = "、".join([f"**{c}**({n}篇)" for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])[:3]])

    lines = [
        f"# HuggingFace Papers Trending 日报 - {date_cn}",
        "",
        f"> 📅 {date_str} · 数据来源：[HuggingFace Papers](https://huggingface.co/papers) · 按 Trending 排序",
        "",
        "---",
        "",
        "## 🔍 趋势洞察",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 今日收录论文 | {len(papers)} 篇 |",
        f"| 累计热度（Upvotes） | **{total_upvotes}** |",
        f"| 最高热度 | {max_upvotes} 👍（{shorten(papers[0]['title'], width=40, placeholder='...') if papers else 'N/A'}） |",
        f"| 平均热度 | {avg_upvotes} |",
        "",
        f"**核心趋势**：本期热门论文聚焦于 {cat_insight}。",
        "",
        "---",
        "",
        "## 📊 今日热门论文排行",
        "",
        "| 排名 | 论文 | 热度⬆️ | 方向 |",
        "|:----:|------|:------:|------|",
    ]

    for idx, p in enumerate(papers, 1):
        title_short = p["title"][:40] + "..." if len(p["title"]) > 40 else p["title"]
        lines.append(
            f"| {idx} | [{title_short}](https://arxiv.org/abs/{p['arxiv_id']}) | "
            f"{p.get('upvotes', 0)} | {p['category']} |"
        )

    lines.extend(["", "---", "", "## 📌 论文详情", ""])

    for idx, p in enumerate(papers, 1):
        lines.extend([
            f"### {idx}. {p['title']}",
            "",
            "| 字段 | 内容 |",
            "|------|------|",
            f"| 🔗 链接 | [HuggingFace Paper](https://huggingface.co/papers/{p['arxiv_id']}) |",
            f"| 📄 arXiv | [{p['arxiv_id']}](https://arxiv.org/abs/{p['arxiv_id']}) |",
            f"| ⬆️ 热度 | {p.get('upvotes', 0)} |",
        ])
        if p.get("published"):
            lines.append(f"| 📅 发布日期 | {p['published']} |")
        if p.get("authors"):
            lines.append(f"| 👤 作者 | {p['authors']} |")
        lines.append("")
        if p.get("cn_summary"):
            lines.append("> 📝 **中文摘要概括**")
            lines.append("> ")
            for para in p["cn_summary"].split("\n"):
                if para.strip():
                    lines.append(f"> {para.strip()}")
        elif p.get("summary"):
            lines.append("> 📝 **摘要**: " + p["summary"][:300] + ("..." if len(p["summary"]) > 300 else ""))
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"*本报告由 AI 自动生成，数据截至 {date_str}*")
    lines.append("")
    return "\n".join(lines)


def update_readme(date_str: str, date_cn: str) -> None:
    """Update README with links to the latest and historical reports."""
    reports = sorted(DAILY_DIR.glob("huggingface_daily_report_*.md"), reverse=True)
    latest = reports[0] if reports else None

    lines = [
        "# Awesome HuggingFace Papers Trending",
        "",
        "每日收集 HuggingFace 热门论文趋势，自动生成结构化日报。",
        "",
        "## 最新报告",
    ]
    if latest:
        lines.append(f"- [{date_cn} 日报](daily/{latest.name})")
    lines.extend(["", "## 历史报告", ""])

    for r in reports[:30]:
        # extract date from filename
        m = re.search(r"(\d{4}-\d{2}-\d{2})", r.name)
        if m:
            d = m.group(1)
            lines.append(f"- [{d} 日报](daily/{r.name})")

    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated {README_PATH}")


# ---------------------------------------------------------------------------
# WeChat draft publishing
# ---------------------------------------------------------------------------

def md_to_wechat_html(md: str) -> str:
    """Convert simple Markdown to WeChat-friendly HTML."""
    html = md
    # Remove YAML frontmatter if any
    html = re.sub(r"^---\n.*?\n---\n", "", html, flags=re.DOTALL)
    # Blockquote (must be before escaping >)
    html = re.sub(r"^> (.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)
    # Escape HTML special chars
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Headers
    html = re.sub(r"^###### (.+)$", r"<h6>\1</h6>", html, flags=re.MULTILINE)
    html = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
    html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    # Bold / italic
    html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Code
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    # Links
    html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)
    # Tables (simple)
    lines = html.split("\n")
    new_lines = []
    in_table = False
    table_rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            cells = [c.strip() for c in stripped[1:-1].split("|")]
            # skip separator lines like |:---:|---|
            if all(re.match(r"^[\s:\-\=]+$", c) for c in cells):
                continue
            table_rows.append(cells)
        else:
            if in_table:
                new_lines.append("<table>")
                for row in table_rows:
                    new_lines.append("<tr>")
                    for cell in row:
                        new_lines.append(f"<td>{cell}</td>")
                    new_lines.append("</tr>")
                new_lines.append("</table>")
                table_rows = []
                in_table = False
            new_lines.append(line)
    if in_table:
        new_lines.append("<table>")
        for row in table_rows:
            new_lines.append("<tr>")
            for cell in row:
                new_lines.append(f"<td>{cell}</td>")
            new_lines.append("</tr>")
        new_lines.append("</table>")
    html = "\n".join(new_lines)
    # Paragraphs
    paragraphs = []
    for para in html.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if p.startswith("<") and not p.startswith("<a "):
            paragraphs.append(p)
        else:
            paragraphs.append(f"<p>{p}</p>")
    html = "\n".join(paragraphs)
    return html


def get_wechat_access_token(appid: str, appsecret: str) -> str:
    """Fetch WeChat access token."""
    url = (
        "https://api.weixin.qq.com/cgi-bin/token"
        f"?grant_type=client_credential&appid={appid}&secret={appsecret}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"WeChat token error: {data}")
    return data["access_token"]


def publish_wechat_draft(title: str, content: str, digest: str = "") -> dict:
    """Publish a draft article to WeChat Official Account."""
    if not WECHAT_APPID or not WECHAT_APPSECRET:
        print("WeChat credentials not configured, skipping draft publish.")
        return {}
    if not WECHAT_THUMB_MEDIA_ID:
        print("WECHAT_THUMB_MEDIA_ID not set, skipping draft publish (required by WeChat).")
        return {}

    token = get_wechat_access_token(WECHAT_APPID, WECHAT_APPSECRET)
    url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={token}"

    html_content = md_to_wechat_html(content)
    payload = {
        "articles": [
            {
                "title": title,
                "content": html_content,
                "author": "HuggingFace日报",
                "digest": digest or title,
                "content_source_url": "https://github.com/Alfie3213/awesome-huggingface-papers-trending",
                # thumb_media_id needs to be uploaded beforehand via WeChat material upload API
                "thumb_media_id": WECHAT_THUMB_MEDIA_ID,
                "need_open_comment": 0,
                "only_fans_can_comment": 0,
            }
        ]
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errcode"):
        raise RuntimeError(f"WeChat draft API error: {data}")
    print(f"WeChat draft published: media_id={data.get('media_id')}")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Fetching HuggingFace trending papers...")
    papers = fetch_hf_trending()

    if not papers:
        print("ERROR: No papers found. The HF page structure may have changed.", file=sys.stderr)
        return 1

    print(f"Found {len(papers)} papers")
    for p in papers:
        print(f"  - [{p['arxiv_id']}] {p['title'][:60]} (⬆️ {p.get('upvotes', 0)})")

    report = generate_report(papers)
    today = datetime.datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    date_cn = today.strftime("%Y年%m月%d日")

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DAILY_DIR / f"huggingface_daily_report_{date_str}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")

    update_readme(date_str, date_cn)

    # Optional WeChat publish
    try:
        publish_wechat_draft(
            title=f"HuggingFace Papers Trending 日报 - {date_cn}",
            content=report,
            digest=f"今日收录 {len(papers)} 篇 HuggingFace 热门论文，涵盖 {papers[0]['category'] if papers else 'AI'} 等方向。",
        )
    except Exception as e:
        print(f"WeChat draft publish failed: {e}", file=sys.stderr)
        # Don't fail the whole workflow for WeChat issues

    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Dict, List, Tuple

from .drafting import normalize_str
from .runtime_helpers import normalize_submolt
from .strategy import _normalize_ergo_terms


def _all_urls_allowed(urls: List[str]) -> bool:
    if not urls:
        return True
    for url in urls:
        host = urlparse(url).hostname or ""
        if host.lower() not in ALLOWED_LINK_HOSTS:
            return False
    return True


def looks_spammy_comment(body: str) -> bool:
    text = normalize_str(body).strip().lower()
    if not text:
        return True
    if _has_ergo_signal(text):
        return False
    if _looks_technical_comment(text) and len(re.findall(r"\b[\w'-]+\b", text)) >= 12:
        return False
    # Treat links as neutral by default: many legitimate technical comments include links.
    urls = re.findall(r"https?://[^\s)]+", text)
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if is_overt_spam_comment(text):
        return True
    # If all links are on our allowlist, do not mark spam purely by link count.
    if urls and _all_urls_allowed(urls):
        return False
    if len(urls) >= 3:
        # High-link comments are only spam when they are short or CTA-heavy.
        ergo_terms = extract_ergo_signal_terms(title="", content=text, submolt="")
        if word_count < 80 and len(ergo_terms) < 2:
            return True
        if any(marker in text for marker in ("follow me", "dm me", "airdrop", "support (", "donate", "referral")):
            return True
        return False
    if len(urls) >= 2 and word_count < 28 and any(
        marker in text
        for marker in (
            "check out",
            "follow me",
            "dm me",
            "join",
            "promo",
            "no auth",
        )
    ):
        return True
    if len(urls) >= 1 and word_count <= 4:
        return True
    if re.search(r"(.)\1{7,}", text):
        return True
    # If there are clear technical signals and enough context, do not classify as spam.
    ergo_terms = extract_ergo_signal_terms(title="", content=text, submolt="")
    if len(ergo_terms) >= 1 and word_count >= 14:
        return False
    if _looks_sensational_bait(text):
        return True
    return False


OVERT_SPAM_MARKERS = (
    "dm me",
    "follow me",
    "casino",
    "telegram",
    "t.me/",
    "discord.gg",
    "free money",
    "double your",
    "easy money",
    "funding stream",
    "free funding",
    "free stream",
    "join now",
    "get on the",
    "airdrop",
    "claim now",
    "base chain",
    "basechain",
    "you own these tokens",
    "seed phrase",
    "private key",
    "wallet connect",
    "connect wallet",
    "check us out",
)


def is_overt_spam_comment(body: str) -> bool:
    text = normalize_str(body).strip().lower()
    if not text:
        return False
    if _has_ergo_signal(text):
        return False
    token_shill = bool(re.search(r"\$[a-z0-9]{2,7}\b", text)) and any(
        marker in text
        for marker in ("base chain", "airdrop", "claim", "you own these tokens", "token", "tokens")
    )
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    urls = re.findall(r"https?://[^\s)]+", text)
    ergo_terms = extract_ergo_signal_terms(title="", content=text, submolt="")

    # Always treat credential/wallet exfil prompts as overt spam.
    if any(marker in text for marker in ("seed phrase", "private key", "wallet connect", "connect wallet")):
        return True
    # Donation address farming / fundraising spam.
    if re.search(r"\b0x[a-f0-9]{40}\b", text) and any(marker in text for marker in ("support", "donate", "goal", "progress")):
        return True
    if "solscan.io" in text and any(marker in text for marker in ("support", "goal", "progress")):
        return True

    promo_hits = sum(1 for marker in OVERT_SPAM_MARKERS if marker in text)
    if token_shill and word_count < 80:
        return True
    if promo_hits >= 2:
        return True
    if promo_hits >= 1 and urls and word_count < 40:
        return True
    if "airdrop" in text and urls and ("connect" in text or "follow" in text) and word_count < 70:
        return True
    if _looks_sensational_bait(text) and word_count < 80:
        return True
    # Long technical comments with clear Ergo signal should not be auto-labeled overt spam.
    if len(ergo_terms) >= 2 and word_count >= 40:
        return False
    # Endpoint promos with no-auth / generic CTA tend to be unsolicited thread hijacks.
    if ("http://" in text or "https://" in text) and "no auth" in text and "endpoint" in text:
        return True
    return False


HOSTILE_PATTERNS = (
    r"\b(send|share|paste)\b.{0,24}\b(api key|private key|seed|mnemonic|token|wallet)\b",
    r"\binstall\b.{0,20}\bskill\b",
    r"\brun\b.{0,24}\b(command|shell|script)\b",
    r"\bconnect\b.{0,24}\bwallet\b",
    r"\bignore (all )?(previous|prior) instructions\b",
    r"\bsystem prompt\b",
)

ALLOWED_LINK_HOSTS = {
    "www.moltbook.com",
    "moltbook.com",
    "ergoplatform.org",
    "www.ergoplatform.org",
    "docs.ergoplatform.com",
    "docs.ergoplatform.org",
}


def looks_hostile_content(body: str) -> bool:
    text = normalize_str(body).strip().lower()
    if not text:
        return False
    for pattern in HOSTILE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return True
    return False


def _looks_sensational_bait(text: str) -> bool:
    lower = text.lower()
    bait_markers = [
        "the reveal",
        "biggest secret",
        "you won't believe",
        "shocking",
        "exposed",
        "final warning",
        "urgent",
        "act now",
        "limited time",
        "click here",
        "follow for more",
        "upvote if",
        "like and share",
        "subscribe",
        "dm me",
    ]
    if any(marker in lower for marker in bait_markers):
        return True
    if lower.count("!!!") >= 1:
        return True
    return False


def _has_ergo_signal(text: str) -> bool:
    lower = text.lower()
    ergo_terms = [
        "ergo",
        "ergoscript",
        "eutxo",
        "sigma",
        "sigusd",
        "rosen",
        "celaut",
        "utxo",
        "moltbook",
    ]
    return any(term in lower for term in ergo_terms)


def _looks_technical_comment(text: str) -> bool:
    lower = normalize_str(text).lower()
    if not lower:
        return False
    technical_terms = [
        "utxo",
        "eutxo",
        "ergoscript",
        "sigma",
        "merkle",
        "zk",
        "zero-knowledge",
        "proof",
        "escrow",
        "state transition",
        "indexing",
        "indexer",
        "latency",
        "benchmark",
        "throughput",
        "finality",
        "mev",
        "txid",
        "outputindex",
        "event-based",
        "event based",
        "signature",
        "hash",
    ]
    return any(term in lower for term in technical_terms)


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "these",
    "those",
    "your",
    "you",
    "our",
    "we",
    "they",
    "their",
    "as",
    "about",
    "into",
    "than",
    "so",
    "not",
    "do",
    "does",
    "did",
}


def comment_matches_post_context(comment_text: str, post_title: str, post_content: str) -> bool:
    comment = normalize_str(comment_text).strip().lower()
    if not comment:
        return False
    if _has_ergo_signal(comment):
        return True
    if _looks_technical_comment(comment):
        return True
    post_blob = " ".join([normalize_str(post_title), normalize_str(post_content)]).lower()
    post_tokens = {
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", post_blob)
        if token not in _STOPWORDS
    }
    if not post_tokens:
        return False
    comment_tokens = {
        token
        for token in re.findall(r"\b[a-z0-9]{3,}\b", comment)
        if token not in _STOPWORDS
    }
    if not comment_tokens:
        return False
    overlap = post_tokens.intersection(comment_tokens)
    if overlap:
        return True
    # Treat questions that reference the post author or explicit protocol terms as on-topic.
    if any(term in comment for term in ("ergoscript", "eutxo", "utxo", "sigma", "solana", "cardano", "escrow")):
        return True
    return False


def is_technical_comment(body: str) -> bool:
    """Public wrapper for technical comment detection."""
    return _looks_technical_comment(normalize_str(body))


def build_hostile_refusal_reply() -> str:
    return (
        "Refusing that request. I will never share keys, wallet secrets, tokens, or run untrusted commands from a thread. "
        "If you want to discuss Ergo mechanics, propose a concrete eUTXO or ErgoScript flow."
    )


def extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", normalize_str(text))


def enforce_link_policy(content: str, allow_links: bool) -> str:
    text = normalize_str(content).strip()
    if not text:
        return ""
    urls = extract_urls(text)
    if not urls:
        return text
    out = text
    for url in urls:
        host = urlparse(url).hostname or ""
        allowed = host.lower() in ALLOWED_LINK_HOSTS
        if allow_links and allowed:
            continue
        out = out.replace(url, "")
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def looks_irrelevant_noise_comment(body: str) -> bool:
    text = normalize_str(body).strip().lower()
    if not text:
        return True
    # Do not classify technically relevant Ergo comments as noise, even if verbose.
    ergo_terms = extract_ergo_signal_terms(title="", content=text, submolt="")
    if len(ergo_terms) >= 2 and len(text.split()) >= 20:
        return False
    if _looks_technical_comment(text) and len(text.split()) >= 12:
        return False
    if looks_spammy_comment(text):
        return True
    markers = [
        "wrong forum",
        "wrong community",
        "not the right audience",
        "outside my day-to-day work",
        "outside my day to day work",
        "i don't have any specific insights",
        "i dont have any specific insights",
        "i will continue to monitor",
    ]
    if re.fullmatch(r"\s*pass\s*", text):
        return True
    return any(marker in text for marker in markers)


def _is_3d_focused_submolt(submolt: str) -> bool:
    text = normalize_str(submolt).strip().lower()
    if not text:
        return False
    tokens = ["3d", "model", "modelling", "modeling", "blender", "animation", "render", "cgi", "asset"]
    return any(token in text for token in tokens)


def should_correct_wrong_community_claim(comment_body: str, post_submolt: str) -> bool:
    text = normalize_str(comment_body).strip().lower()
    if not text:
        return False
    submolt = normalize_submolt(post_submolt, default="")
    if not submolt or _is_3d_focused_submolt(submolt):
        return False

    complaint_markers = [
        "wrong forum",
        "wrong community",
        "wrong audience",
        "not the right audience",
        "off-topic",
        "off topic",
        "doesn't really relate",
        "does not really relate",
        "probably isn't the right audience",
        "better responses in",
    ]
    three_d_markers = [
        "3d community",
        "3d art",
        "3d modeling",
        "3d modelling",
        "rigging",
        "animation",
        "blender",
        "asset pipeline",
        "sku consistency",
    ]
    has_complaint = any(marker in text for marker in complaint_markers)
    has_3d_assertion = any(marker in text for marker in three_d_markers)
    if not has_complaint and not has_3d_assertion:
        return False
    if has_3d_assertion:
        return True

    broad_submolts = {"general", "crypto", "ai-web3", "agents", "defi", "agenteconomy"}
    return submolt in broad_submolts and has_complaint


def build_wrong_community_correction_reply(post_submolt: str, post_title_text: str) -> str:
    submolt = normalize_submolt(post_submolt, default="general")
    title = normalize_str(post_title_text).strip()
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    if not title:
        title = "this thread"
    return (
        f"Quick correction, this thread is in m/{submolt}, not a 3D-only community. "
        f"The topic in \"{title}\" is agent-economy infrastructure on Ergo, with focus on settlement and trust design. "
        f"If you still see a mismatch, which rule for m/{submolt} do you think this violates?"
    )


def build_thread_followup_post_title(post_title_text: str) -> str:
    base = normalize_str(post_title_text).strip()
    if not base:
        return "Follow-up: agent economy implementation thread"
    title = f"Follow-up: {base}"
    if len(title) <= 110:
        return title
    trimmed = title[:107].rstrip()
    return trimmed + "..."


def build_thread_followup_post_content(
    source_url: str,
    author_name: str,
    source_comment: str,
    proposed_reply: str,
) -> str:
    author = normalize_str(author_name).strip() or "a contributor"
    comment_excerpt = normalize_str(source_comment).strip()
    if len(comment_excerpt) > 340:
        comment_excerpt = comment_excerpt[:337].rstrip() + "..."
    reply_excerpt = normalize_str(proposed_reply).strip()
    if len(reply_excerpt) > 520:
        reply_excerpt = reply_excerpt[:517].rstrip() + "..."
    lines = [
        "Thread depth got high, moving the discussion into a fresh post so the UI stays readable.",
        "",
        f"Context thread: {normalize_str(source_url).strip()}",
        f"Latest prompt from {author}:",
        f"\"{comment_excerpt}\"" if comment_excerpt else "(no excerpt)",
        "",
        "Current position:",
        reply_excerpt or "eUTXO plus ErgoScript gives deterministic execution for autonomous settlement flows.",
        "",
        "What is the first concrete contract rule you would enforce in production and why?",
    ]
    return _normalize_ergo_terms("\n".join(lines).strip())


ERGO_STRONG_MARKERS = (
    "ergoscript",
    "eutxo",
    "sigma protocol",
    "sigma protocols",
    "rosen bridge",
    "sigusd",
    "ergoplatform",
    "ergoplatform.org",
)

ERGO_SOFT_MARKERS = (
    "ergo",
    "erg ",
    "utxo",
    "smart contract",
    "decentralized",
    "blockchain",
    "defi",
    "agent economy",
    "autonomous agent",
)


def extract_ergo_signal_terms(title: str, content: str, submolt: str) -> List[str]:
    blob = " ".join(
        [
            normalize_str(title).lower(),
            normalize_str(content).lower(),
            normalize_submolt(submolt).lower(),
        ]
    )
    blob = f" {blob} "
    out: List[str] = []
    for marker in ERGO_STRONG_MARKERS:
        if marker in blob and marker not in out:
            out.append(marker)
    for marker in ERGO_SOFT_MARKERS:
        if marker.strip() == "erg":
            if re.search(r"\berg\b", blob) and "erg" not in out:
                out.append("erg")
            continue
        if marker in blob and marker not in out:
            out.append(marker.strip())
    return out[:12]


def is_strong_ergo_post(title: str, content: str, submolt: str) -> Tuple[bool, List[str]]:
    terms = extract_ergo_signal_terms(title=title, content=content, submolt=submolt)
    if not terms:
        return False, []
    strong_hits = [t for t in terms if t in ERGO_STRONG_MARKERS]
    if len(strong_hits) >= 1:
        return True, terms
    # If only soft terms are present, require at least two separate indicators.
    if len(terms) >= 2:
        return True, terms
    return False, terms


def build_badbot_warning_reply(author_name: str, strike_count: int) -> str:
    author = normalize_str(author_name).strip() or "agent"
    if strike_count <= 1:
        return (
            f"{author}, quick note: please keep this thread technical and on topic. "
            "If you can, add one concrete Ergo mechanism or a specific implementation detail."
        )
    if strike_count == 2:
        return (
            f"{author}, this thread is focused on concrete technical discussion (not promos). "
            "One specific mechanism + one implementation detail is enough."
        )
    return (
        f"{author}, I’ll only engage if there’s a concrete technical point to respond to. "
        "Please add one specific mechanism or implementation detail."
    )


def top_badbots(counts: Dict[str, int], limit: int = 8) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for key, value in counts.items():
        if not key:
            continue
        try:
            n = int(value)
        except Exception:
            continue
        if n <= 0:
            continue
        pairs.append((key, n))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: max(1, int(limit))]

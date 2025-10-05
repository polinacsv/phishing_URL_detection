import pandas as pd
import numpy as np
import tldextract
from typing import Set
import re
from urllib.parse import urlparse, parse_qs  # 



def extract_mld(url: str) -> str:
    """Extract main-level domain + public suffix (e.g. example.co.uk)."""
    if not isinstance(url, str) or not url:
        return ""
    u = url if "://" in url else f"http://{url}"
    ext = tldextract.extract(u)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return ""


def add_alexa_features(df: pd.DataFrame, alexa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - mld_ps (main-level domain + public suffix)
      - is_in_alexa (1 if domain is in Alexa Top list, else 0)
      - ranking (Alexa rank, 1_000_001 if not in Alexa)
    Expects alexa_df columns: 'alexa_domain', 'ranking'
    """
    import tldextract

    def extract_mld(url: str) -> str:
        if not isinstance(url, str) or not url:
            return ""
        u = url if "://" in url else f"http://{url}"
        ext = tldextract.extract(u)
        return f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else ""

    out = df.copy()
    out["mld_ps"] = out["url"].apply(extract_mld)

    merged = out.merge(
        alexa_df[["alexa_domain", "ranking"]],
        how="left",
        left_on="mld_ps",
        right_on="alexa_domain",
    )

    out["is_in_alexa"] = merged["ranking"].notna().astype(int)
    out["ranking"] = merged["ranking"].fillna(1_000_001).astype(int)
    return out


PHISHY_TOKENS = {
    "login","signin","verify","update","secure","security","confirm","account",
    "webscr","passcode","password","credential","unlock","bank","wallet","invoice",
    "checkout","reset","support","appeal","limit","suspend","pay","paypal"
}
COMMON_SUBDOMAINS = {"www","m","mail","docs","drive","calendar","api","cdn","img","static"}

URL_SHORTENERS = {
    "bit.ly","tinyurl.com","t.co","goo.gl","is.gd","v.gd","ow.ly","buff.ly",
    "rebrand.ly","cutt.ly","shorturl.at","lnkd.in","bit.do","t.ly"
}

def _mld(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    u = url if "://" in url else f"http://{url}"
    ext = tldextract.extract(u)
    return f"{ext.domain}.{ext.suffix}".lower() if ext.domain and ext.suffix else ""

def add_suspicious_subdomain_or_path(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mld_ps" not in out.columns:
        out["mld_ps"] = out["url"].apply(_mld)

    def feat_row(u: str):
        if not isinstance(u, str) or not u:
            return 0
        u2 = u if "://" in u else f"http://{u}"
        p = urlparse(u2)
        path = (p.path or "").lower()
        q = parse_qs(p.query or "")   

        # subdomain heuristics (via tldextract)
        ext = tldextract.extract(u2)
        sub = ext.subdomain.lower() if ext.subdomain else ""
        sub_parts = [s for s in sub.split(".") if s]
        sub_depth = len(sub_parts)
        alnum = re.sub(r"[^a-z0-9]", "", sub)
        digit_ratio = (sum(c.isdigit() for c in alnum) / max(1, len(alnum)))
        long_sub = len(sub) > 25
        deep_sub = sub_depth >= 2 and not set(sub_parts).issubset(COMMON_SUBDOMAINS)
        randomish = digit_ratio > 0.30 or long_sub

        # path/query heuristics
        path_len = len(path)
        many_params = len(q) >= 4
        has_token = any(tok in path for tok in PHISHY_TOKENS)

        return int(deep_sub or randomish or has_token or path_len > 60 or many_params)

    out["suspicious_subdomain_or_path"] = out["url"].astype(str).map(feat_row)
    return out

def add_is_url_shortener(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mld_ps" not in out.columns:
        out["mld_ps"] = out["url"].apply(_mld)
    out["is_url_shortener"] = out["mld_ps"].isin(URL_SHORTENERS).astype(int)
    return out


def add_alexa_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transformed Alexa features for modeling:
      - log_ranking: log1p(ranking) to compress skewed distribution
      - keep is_in_alexa as-is
      - drop raw 'ranking' to avoid redundancy
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['is_in_alexa', 'ranking']
    
    Returns
    -------
    pd.DataFrame
        Copy of df with log_ranking added and raw ranking dropped.
    """
    if "ranking" not in df.columns or "is_in_alexa" not in df.columns:
        raise ValueError("Input df must contain 'ranking' and 'is_in_alexa'")
    
    out = df.copy()
    out["log_ranking"] = np.log1p(out["ranking"])
    
    # Drop raw ranking since log version replaces it
    out = out.drop(columns=["ranking"])
    
    return out

def tokenize(text: str) -> Set[str]:
    """
    Splits input text into lowercase alphanumeric tokens.
    Useful for comparing similarity between URL parts.

    Parameters:
        text (str): Input string (e.g., a URL path or domain)

    Returns:
        Set[str]: Set of lowercase alphanumeric tokens
    """
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Computes Jaccard similarity between two sets:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Returns:
        float: Jaccard similarity score (0.0 to 1.0)
    """
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def extract_jaccard_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Jaccard similarity features comparing token overlap between:
      - mld_ps and path                    -> jaccard_mld_path
      - subdomain and path                 -> jaccard_subdomain_path
      - hostname (sub+domain) and path     -> jaccard_hostname_path
      - mld_ps and query tokens            -> jaccard_mld_query
      - subdomain and query tokens         -> jaccard_subdomain_query
      - path tokens and query tokens       -> jaccard_path_query
    Also adds:
      - path_self_similarity: repetition ratio of most common path token

    Requires: columns ['url', 'mld_ps'] and helpers tokenize(), jaccard_similarity().
    """
    df = df.copy()

    # Ensure URLs are parseable by adding scheme if missing
    df['full_url'] = df['url'].apply(lambda u: u if isinstance(u, str) and u.startswith('http') else f'http://{u}')

    def compute_all(url, mld_ps):
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""

            # Token sets
            path_tokens = tokenize(parsed.path)
            mld_tokens = tokenize(mld_ps) if pd.notnull(mld_ps) else set()

            parts = hostname.split('.') if hostname else []
            subdomain_parts = parts[:-2] if len(parts) > 2 else []
            subdomain_tokens = tokenize('.'.join(subdomain_parts))

            hostname_tokens = tokenize(hostname)

            # Query tokens (keys + values)
            q = parse_qs(parsed.query or "")
            query_tokens = set()
            for k, vals in q.items():
                query_tokens |= tokenize(k)
                for v in vals:
                    query_tokens |= tokenize(v)

            # Jaccards
            j_mld_path  = jaccard_similarity(mld_tokens, path_tokens)
            j_sub_path  = jaccard_similarity(subdomain_tokens, path_tokens)
            j_host_path = jaccard_similarity(hostname_tokens, path_tokens)
            j_mld_query = jaccard_similarity(mld_tokens, query_tokens)
            j_sub_query = jaccard_similarity(subdomain_tokens, query_tokens)
            j_path_query = jaccard_similarity(path_tokens, query_tokens)

            # Path self-similarity (token repetition in path)
            if path_tokens:
                from collections import Counter
                c = Counter(path_tokens)
                path_self = max(c.values()) / max(1, sum(c.values()))
            else:
                path_self = 0.0

            return pd.Series([
                j_mld_path, j_sub_path, j_host_path,
                j_mld_query, j_sub_query, j_path_query,
                path_self
            ])
        except Exception:
            return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    df[[
        'jaccard_mld_path',
        'jaccard_subdomain_path',
        'jaccard_hostname_path',
        'jaccard_mld_query',
        'jaccard_subdomain_query',
        'jaccard_path_query',
        'path_self_similarity'
    ]] = df.apply(lambda row: compute_all(row['full_url'], row['mld_ps']), axis=1)

    return df



def compute_shannon_entropy(text: str) -> float:
    """
    Computes Shannon entropy of a string, which measures the randomness
    or complexity of the characters.

    Parameters:
        text (str): Input string (typically a URL or part of it)

    Returns:
        float: Shannon entropy value
    """
    if not text:
        return 0.0

    from collections import Counter
    counts = Counter(text)
    probabilities = [count / len(text) for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy


def add_entropy_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a Shannon entropy feature to the DataFrame based on the full URL.

    Parameters:
        df (pd.DataFrame): Must contain a 'url' column

    Returns:
        pd.DataFrame: DataFrame with new column:
                      - 'url_entropy': Shannon entropy of the full URL
    """
    df = df.copy()

    # Ensure full URL has a scheme for consistent parsing
    df['full_url'] = df['url'].apply(lambda u: u if u.startswith('http') else f'http://{u}')

    df['url_entropy'] = df['full_url'].apply(compute_shannon_entropy)
    return df




def add_url_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds structural URL features:
    - num_digits
    - url_length
    - num_subdomains
    - num_path_segments
    - num_suspicious_keywords
    - num_special_chars
    - has_brand_conflict
    """
    df = df.copy()

    df['full_url'] = df['url'].apply(lambda u: u if u.startswith('http') else f'http://{u}')
    
    suspicious_keywords = {'login', 'secure', 'verify', 'account', 'update', 'banking',
                           'webscr', 'ebay', 'paypal', 'signin', 'submit', 'confirm'}

    known_brands = {'paypal', 'apple', 'amazon', 'google', 'facebook', 'bankofamerica', 'microsoft'}

    def extract_structure_features(url: str):
        try:
            parsed = urlparse(url)
            hostname_parts = parsed.hostname.split('.') if parsed.hostname else []

            num_digits = sum(c.isdigit() for c in url)
            url_length = len(url)
            num_subdomains = max(0, len(hostname_parts) - 2)
            num_path_segments = len(parsed.path.strip('/').split('/')) if parsed.path else 0
            url_tokens = re.findall(r"[a-zA-Z]+", url.lower())
            num_suspicious_keywords = sum(token in suspicious_keywords for token in url_tokens)
            num_special_chars = len(re.findall(r"[^a-zA-Z0-9]", url))

            # brand conflict check
            domain_part = ".".join(hostname_parts[-2:]) if len(hostname_parts) >= 2 else parsed.hostname
            path_part = parsed.path.lower()
            domain_brands = [b for b in known_brands if b in domain_part]
            path_brands = [b for b in known_brands if b in path_part]
            has_brand_conflict = int(
                len(domain_brands) > 0 and
                any(b not in domain_brands for b in path_brands)
            )

            return pd.Series([
                num_digits,
                url_length,
                num_subdomains,
                num_path_segments,
                num_suspicious_keywords,
                num_special_chars,
                has_brand_conflict
            ])
        except:
            return pd.Series([0, 0, 0, 0, 0, 0, False])

    df[[
        'num_digits', 'url_length', 'num_subdomains',
        'num_path_segments', 'num_suspicious_keywords',
        'num_special_chars', 'has_brand_conflict'
    ]] = df['full_url'].apply(extract_structure_features)

    return df


_ALNUM = re.compile(r"[A-Za-z0-9]+")

# ---- deterministic, no-hashlib hash: FNV-1a 32-bit ----
def _fnv1a_32(s: str) -> int:
    h = 0x811C9DC5  # offset basis
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, keep 32-bit
    return h

def _idx(s: str, dim: int = 256) -> int:
    return _fnv1a_32(s) % dim

def _ensure_http(u: str) -> str:
    return u if isinstance(u, str) and re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u) else (f"http://{u}" if isinstance(u, str) else "http://")

def _tok_host(p):
    return _ALNUM.findall((p.hostname or "").lower())

def _tok_path(p):
    return _ALNUM.findall((p.path or "").lower())

def _hash_trigram_vec(tokens, dim: int = 256) -> np.ndarray:
    if not tokens:
        return np.zeros(dim)
    v = np.zeros(dim, dtype=float)
    s = f"^{ ' '.join(tokens) }$"
    L = len(s)
    if L >= 3:
        for i in range(L - 2):
            tri = s[i:i+3]
            v[_idx(tri, dim)] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v

def add_token_level_features_offline(df: pd.DataFrame, *, dim: int = 256) -> pd.DataFrame:
    """
    Adds three offline features (no hashlib, no internet):
      - domain_word2vec_similarity  (cosine of hashed trigram vectors for domain vs path)
      - avg_token_length            (mean length over domain+path tokens)
      - digit_letter_ratio          (digits / letters over domain+path tokens)
    """
    df = df.copy()
    if "full_url" not in df.columns:
        df["full_url"] = df["url"].map(_ensure_http)

    def _compute(u: str):
        p = urlparse(u)
        dom = _tok_host(p)
        path = _tok_path(p)

        v_dom  = _hash_trigram_vec(dom,  dim=dim)
        v_path = _hash_trigram_vec(path, dim=dim)
        # cosine similarity (0 if one side is empty)
        sim = float(np.dot(v_dom, v_path)) if (v_dom.any() and v_path.any()) else 0.0

        toks = dom + path
        if not toks:
            toks = _ALNUM.findall((p.geturl() or "").lower())

        avg_len = float(np.mean([len(t) for t in toks])) if toks else 0.0
        text = "".join(toks)
        letters = sum(c.isalpha() for c in text)
        digits  = sum(c.isdigit() for c in text)
        ratio   = digits / (letters if letters else 1)

        return pd.Series([sim, avg_len, ratio])

    cols = ["domain_word2vec_similarity", "avg_token_length", "digit_letter_ratio"]
    df[cols] = df["full_url"].apply(_compute)
    return df
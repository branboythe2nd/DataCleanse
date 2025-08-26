#!/usr/bin/env python3
# entity_clusterer.py
# Human-in-the-loop entity-equivalence clustering with automatic helper+acronym merging.
#
# Example:
#   python entity_clusterer.py \
#     --input_dir ./data \
#     --output_dir ./out \
#     --feedback_dir ./feedback \
#     --name_cols "name,company,supplier,entity" \
#     --helper_cols "country,zip,domain,email,website,phone,hs_code,vendor_id,tax_id" \
#     --id_col id
#
# Workflow:
# 1) First run writes proposals: feedback/proposals_round_01.csv
#    - Review feedback/approvals_round_01.csv (includes readable "items") and mark decision=approve for clusters to freeze.
# 2) Run again: the script reads approvals, treats approved clusters as *must-link seeds* (they won’t split),
#    then auto-merges components using helper keys + acronym/initialism rules. You still only approve/disapprove.
# 3) Repeat until satisfied. Current mapping is out/clusters_latest.csv; state is out/state.json.
#
# NOTE: No constraints.csv. Merging happens automatically based on helpers & acronyms.

import argparse, glob, json, math, os, random, re, sys, hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Iterable
from collections import defaultdict

import numpy as np
import pandas as pd

# -------- Optional deps handling (embeddings are optional) ----------
HAVE_SENTENCE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    HAVE_SENTENCE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

# RapidFuzz for string similarity (fast & robust)
HAVE_RAPIDFUZZ = True
try:
    from rapidfuzz import fuzz
except Exception:
    HAVE_RAPIDFUZZ = False

# ---------------- Normalization helpers -----------------
PUNCT_RE = re.compile(r"[^\w\s\-/&]")
MULTI_SPACE = re.compile(r"\s+")

# Suffixes/abbrev tuned for org names
CORP_SUFFIXES = {
    "inc","inc.","llc","l.l.c.","ltd","ltd.","co","co.","corp","corp.","corporation",
    "company","s.a.","sa","ag","gmbh","pte","plc","bv","oy","oyj","ab","sarl","nv",
    "kk","aps","as","kft","sro","s.r.o","sp","sp.","sp. z o.o.","sp z oo"
}
ABBR = {"intl":"international", "int'l":"international", "univ":"university",
        "tech":"technology", "mfg":"manufacturing", "dept":"department"}

GENERIC_EMAIL_DOMAINS = {
    "gmail.com","yahoo.com","outlook.com","hotmail.com","aol.com","icloud.com",
    "live.com","msn.com","proton.me","protonmail.com","yandex.ru","qq.com","163.com","126.com"
}

def strip_accents(s: str) -> str:
    try:
        import unicodedata as ud
        return ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return s

def canon_header(col: str) -> str:
    c = col.strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = strip_accents(str(s).lower().strip())
    s = PUNCT_RE.sub(" ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        if t in CORP_SUFFIXES:  # drop corporate suffixes
            continue
        out.append(ABBR.get(t, t))
    return " ".join(out)

def tokenize_for_set(s: str) -> List[str]:
    s = normalize_text(s)
    toks = [t for t in s.split() if t and t not in {"and","the","of","for","at","to","in","a","an"}]
    return sorted(set(toks))

def etld1_from_value(val: str) -> Optional[str]:
    if not val: return None
    v = str(val).strip().lower()
    # extract domain from email/URL
    if "@" in v:
        v = v.split("@",1)[1]
    v = re.sub(r"^https?://", "", v)
    v = v.split("/")[0]
    parts = [p for p in v.split(".") if p]
    if not parts:
        return None
    dom = ".".join(parts[-2:]) if len(parts) >= 2 else parts[0]
    if dom in GENERIC_EMAIL_DOMAINS:
        return None
    return dom

def norm_phone(val: str) -> Optional[str]:
    if not val: return None
    ds = re.sub(r"\D", "", str(val))
    if len(ds) < 7:
        return None
    return ds[-10:]

def norm_zip(val: str) -> Optional[str]:
    if not val: return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(val)).upper()
    return s or None

def norm_country(val: str) -> Optional[str]:
    if not val: return None
    s = normalize_text(val).upper()
    return s or None

def norm_taxid(val: str) -> Optional[str]:
    if not val: return None
    s = re.sub(r"[^A-Za-z0-9]", "", str(val)).upper()
    if len(s) < 6:
        return None
    return s

def norm_vendorid(val: str) -> Optional[str]:
    if not val: return None
    s = re.sub(r"\s+", "", str(val)).upper()
    if len(s) < 3:
        return None
    return s

# ---------------- Similarity primitives -----------------
def rf_string_similarity(a: str, b: str) -> float:
    if not HAVE_RAPIDFUZZ:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    ts = fuzz.token_set_ratio(a, b) / 100.0
    pr = fuzz.partial_ratio(a, b) / 100.0
    wr = fuzz.WRatio(a, b) / 100.0
    return 0.5*ts + 0.2*pr + 0.3*wr

# --- Acronym / initialism helpers ---
STOP_TOKENS = {"and","the","of","for","at","to","in","a","an"}
_ACR_CLEAN_RE = re.compile(r"[^A-Za-z]")

def clean_acronym(s: str) -> str:
    return _ACR_CLEAN_RE.sub("", s or "").upper()

def is_pure_acronym_raw(raw: str) -> bool:
    if not raw:
        return False
    s = clean_acronym(raw)
    if not (2 <= len(s) <= 12):
        return False
    if " " in str(raw).strip():
        return False
    letters_only = re.sub(r"[^A-Za-z]", "", str(raw))
    return letters_only.upper() == s

def initialism_from_norm(norm_name: str) -> str:
    if not norm_name:
        return ""
    toks = [t for t in norm_name.split() if t and t not in STOP_TOKENS and t not in CORP_SUFFIXES]
    letters = [t[0] for t in toks if t and t[0].isalpha()]
    return "".join(letters).upper()

_PAREN_PAT = re.compile(r"^(?P<long>.+?)\s*\((?P<acr>[A-Za-z&\.]{2,12})\)\s*$")
_DASH_PAT  = re.compile(r"^(?P<acr>[A-Za-z&\.]{2,12})\s*[-–—]\s*(?P<long>.+)$")

def build_alias_map(raw_names: List[str], norm_names: List[str]) -> Dict[str, set]:
    alias = defaultdict(set)
    for raw, norm in zip(raw_names, norm_names):
        s = (raw or "").strip()
        m = _PAREN_PAT.match(s) or _DASH_PAT.match(s)
        if not m:
            continue
        long_raw = m.group("long").strip()
        acr_raw  = m.group("acr").strip()
        long_norm = normalize_text(long_raw)
        acr_clean = clean_acronym(acr_raw)
        if not long_norm or not acr_clean:
            continue
        if acr_clean == initialism_from_norm(long_norm):
            alias[long_norm].add(acr_clean)
            alias[acr_clean].add(long_norm)
    return alias

def is_acronym_of_clean(acr_clean: str, long_norm: str) -> bool:
    return bool(acr_clean) and acr_clean == initialism_from_norm(long_norm)

# ---------------- Blocking keys -----------------
def soundex_simple(s: str) -> str:
    s = re.sub(r"[^a-z]", "", s.split()[0]) if s else ""
    if not s: return ""
    first = s[0]
    mapping = {"bfpv":"1","cgjkqsxz":"2","dt":"3","l":"4","mn":"5","r":"6"}
    codes = []
    for ch in s[1:]:
        code = ""
        for k,v in mapping.items():
            if ch in k:
                code = v
                break
        if code and (not codes or codes[-1] != code):
            codes.append(code)
        if len(codes) == 3: break
    return first + "".join(codes).ljust(3,"0")

def block_keys(norm_name: str, init_cap: int = 6) -> Iterable[str]:
    if not norm_name:
        return []
    toks = norm_name.split()
    keys = set()
    if toks:
        keys.add(f"head3:{toks[0][:3]}")
        keys.add(f"sx:{soundex_simple(norm_name)}")
    if len(norm_name) >= 3:
        keys.add(f"tri:{norm_name[:3]}")
    keys.add(f"len:{len(norm_name)//3}")
    init = initialism_from_norm(norm_name)
    if len(init) >= 2:
        keys.add(f"init:{init[:init_cap]}")
    return keys

# ---------------- State -----------------
@dataclass
class ClusterState:
    round: int = 1
    approved_clusters: List[List[str]] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)

def load_state(path: str) -> ClusterState:
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            d = json.load(f)
        return ClusterState(**d)
    return ClusterState()

def save_state(path: str, st: ClusterState):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(asdict(st), f, indent=2)

# ---------------- DSU -----------------
class DSU:
    def __init__(self, n:int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# ---------------- Feature cache per run -----------------
class FeatureCache:
    def __init__(self, names: List[str], df: pd.DataFrame, helper_cols: List[str]):
        self.names = names
        self.norm = [normalize_text(n) for n in names]
        self.tokens = [set(tokenize_for_set(n)) for n in names]
        self.len_arr = np.array([len(s) for s in self.norm], dtype=float)

        # Acronym flags
        self.acr_clean = [clean_acronym(n) for n in names]
        self.is_acronym = [is_pure_acronym_raw(n) for n in names]

        # Char TF-IDF
        self.tfv = TfidfVectorizer(analyzer="char", ngram_range=(3,5), lowercase=True, min_df=1, norm="l2")
        self.Xc = self.tfv.fit_transform(self.norm)

        # Optional sentence embeddings
        self.embs = None
        if HAVE_SENTENCE:
            try:
                model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                embs = model.encode(self.norm, normalize_embeddings=True, show_progress_bar=False)
                self.embs = np.asarray(embs)
            except Exception:
                self.embs = None

        # Alias map like 'Long (LNG)' / 'LNG - Long'
        self.alias_map = build_alias_map(self.names, self.norm)

        # Helper keys per row + buckets for blocking and merging
        self.helper_keys: List[set] = [set() for _ in range(len(df))]
        self.helper_buckets: Dict[str, List[int]] = defaultdict(list)
        self._build_helper_keys(df, helper_cols)

    def _build_helper_keys(self, df: pd.DataFrame, helper_cols: List[str]):
        for i in range(len(df)):
            keys = set()
            row = df.iloc[i].to_dict()
            for col in helper_cols:
                v = row.get(col, None)
                if v is None or (isinstance(v, str) and not v.strip()): 
                    continue
                cl = col.lower()
                # country (weak key, used in combination)
                if ("country" in cl) or (cl == "ctry") or ("iso" in cl):
                    c = norm_country(str(v))
                    if c:
                        keys.add(f"ctry:{c}")
                # domain/email/website/url
                if ("domain" in cl) or ("email" in cl) or ("website" in cl) or ("url" in cl):
                    d = etld1_from_value(str(v))
                    if d:
                        keys.add(f"dom:{d}")
                # phone
                if "phone" in cl or "tel" in cl or "mobile" in cl:
                    p = norm_phone(str(v))
                    if p:
                        keys.add(f"phone:{p}")
                # zip/postal
                if "zip" in cl or "postal" in cl or "postcode" in cl:
                    z = norm_zip(str(v))
                    if z:
                        c = None
                        for cc, vv in row.items():
                            cc_l = str(cc).lower()
                            if "country" in cc_l or cc_l == "ctry" or "iso" in cc_l:
                                c = norm_country(vv); break
                        if c:
                            keys.add(f"zip:{c}:{z}")
                        else:
                            keys.add(f"zip::{z}")
                # HS code
                if "hs" in cl and "code" in cl:
                    hs = re.sub(r"\D", "", str(v))[:8]
                    if hs:
                        keys.add(f"hs:{hs}")
                # tax id
                if "tax" in cl or "ein" in cl or "tin" in cl or "vat" in cl or "gst" in cl:
                    tid = norm_taxid(str(v))
                    if tid:
                        keys.add(f"tax:{tid}")
                # vendor/customer/supplier ids
                if "vendor" in cl or "supplier" in cl or "customer" in cl or "client" in cl or "partner" in cl or cl.endswith("_id"):
                    vid = norm_vendorid(str(v))
                    if vid:
                        keys.add(f"vid:{vid}")

            self.helper_keys[i] = keys
            for k in keys:
                self.helper_buckets[k].append(i)

# --------------- Candidate generation (blocked, capped) ---------------
def generate_blocked_slices(cache: FeatureCache, cap:int=5000) -> List[List[int]]:
    """
    Buckets:
      - name-derived keys (soundex, prefixes, initialism)
      - explicit acronym bucket for pure acronyms
      - helper-derived keys (domain, phone, country, zip+country, tax/vendor/hs)
    """
    buckets = defaultdict(list)
    for i, norm in enumerate(cache.norm):
        for k in block_keys(norm):
            buckets[k].append(i)
        if cache.is_acronym[i]:
            buckets[f"acr:{cache.acr_clean[i]}"].append(i)
        for hk in cache.helper_keys[i]:
            buckets[hk].append(i)

    seen_hashes = set()
    slices = []
    for _, idxs in buckets.items():
        idxs = sorted(set(idxs))
        if not idxs:
            continue
        if len(idxs) > cap:
            subb = defaultdict(list)
            for j in idxs:
                toks = cache.norm[j].split()
                subkey = toks[-1][0] if toks else "#"
                subb[subkey].append(j)
            for sub in subb.values():
                h = hashlib.md5((",".join(map(str,sorted(sub)))).encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h); slices.append(sorted(sub))
        else:
            h = hashlib.md5((",".join(map(str,idxs))).encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h); slices.append(idxs)
    return slices

def knn_pairs_sparse(X, k:int) -> List[Tuple[int,int,float]]:
    n = X.shape[0]
    if n <= 1: return []
    k = max(2, min(k, n))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    out = []
    for i in range(n):
        for r in range(1, k):
            j = int(idxs[i, r])
            if i < j:
                out.append((i, j, 1.0 - float(dists[i, r])))
    return out

def knn_pairs_dense(embs: np.ndarray, k:int) -> List[Tuple[int,int,float]]:
    n = embs.shape[0]
    if n <= 1: return []
    k = max(2, min(k, n))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(embs, return_distance=True)
    out = []
    for i in range(n):
        for r in range(1, k):
            j = int(idxs[i, r])
            if i < j:
                out.append((i, j, 1.0 - float(dists[i, r])))
    return out

# --------------- Pair features & scoring ----------------
def helper_similarity(irow: dict, jrow: dict, helper_cols: List[str]) -> Tuple[float,int,float]:
    # Returns (avg_match, count_contributing_cols, max_signal)
    if not helper_cols: return (0.0, 0, 0.0)
    hits = []
    maxsig = 0.0
    for col in helper_cols:
        vi, vj = irow.get(col, ""), jrow.get(col, "")
        if not (isinstance(vi, str) or isinstance(vj, str)):
            vi, vj = str(vi), str(vj)
        ni, nj = normalize_text(vi), normalize_text(vj)
        if not ni or not nj:
            continue
        score = 1.0 if ni == nj else 0.0
        cl = col.lower()
        # Domain-aware
        if ("domain" in cl) or ("email" in cl) or ("website" in cl) or ("url" in cl):
            di, dj = etld1_from_value(vi), etld1_from_value(vj)
            if di and dj:
                score = 1.0 if di == dj else 0.0
        # Phone
        if ("phone" in cl) or ("tel" in cl) or ("mobile" in cl):
            pi, pj = norm_phone(vi), norm_phone(vj)
            if pi and pj:
                score = 1.0 if pi == pj else 0.0
        # Zip/postal (weak by itself)
        if ("zip" in cl) or ("postal" in cl) or ("postcode" in cl):
            zi, zj = norm_zip(vi), norm_zip(vj)
            if zi and zj:
                score = 1.0 if zi == zj else 0.0
        # Tax IDs
        if ("tax" in cl) or ("ein" in cl) or ("tin" in cl) or ("vat" in cl) or ("gst" in cl):
            ti, tj = norm_taxid(vi), norm_taxid(vj)
            if ti and tj:
                score = 1.0 if ti == tj else 0.0
        # Vendor/Customer IDs
        if ("vendor" in cl) or ("supplier" in cl) or ("customer" in cl) or ("client" in cl) or ("partner" in cl) or cl.endswith("_id"):
            vi2, vj2 = norm_vendorid(vi), norm_vendorid(vj)
            if vi2 and vj2:
                score = 1.0 if vi2 == vj2 else 0.0
        # Country (weak by itself)
        if ("country" in cl) or (cl == "ctry") or ("iso" in cl):
            ci, cj = norm_country(vi), norm_country(vj)
            if ci and cj:
                score = 1.0 if ci == cj else 0.0

        hits.append(score)
        maxsig = max(maxsig, score)
    if not hits: return (0.0, 0, 0.0)
    return (float(sum(hits)/len(hits)), len(hits), maxsig)

def acronym_similarity(a_norm: str, b_norm: str,
                       a_is_acr: bool, b_is_acr: bool,
                       a_acr_clean: str, b_acr_clean: str,
                       alias_map: Dict[str, set]) -> float:
    # Strong if exactly acronym <-> longform
    if a_is_acr and not b_is_acr and is_acronym_of_clean(a_acr_clean, b_norm):
        return 1.0
    if b_is_acr and not a_is_acr and is_acronym_of_clean(b_acr_clean, a_norm):
        return 1.0
    # Learned aliases
    if (a_norm in alias_map and b_norm in alias_map[a_norm]) or \
       (b_norm in alias_map and a_norm in alias_map[b_norm]):
        if (a_is_acr and not b_is_acr) or (b_is_acr and not a_is_acr):
            return 1.0
    # Weak: same initialism, ONLY when neither side is a pure acronym
    if not a_is_acr and not b_is_acr:
        ia = initialism_from_norm(a_norm); ib = initialism_from_norm(b_norm)
        if ia and ib and ia == ib and len(ia) >= 2:
            return 0.6
    return 0.0

def pair_features(i:int, j:int, cache: FeatureCache, df: pd.DataFrame, helper_cols: List[str]) -> np.ndarray:
    # 8D feature vector
    ni, nj = cache.norm[i], cache.norm[j]
    rf = rf_string_similarity(ni, nj)

    si, sj = cache.tokens[i], cache.tokens[j]
    jacc = (len(si & sj) / len(si | sj)) if (si or sj) else 1.0
    overlap = float(len(si & sj))

    chr_cos = 0.0
    Xi, Xj = cache.Xc[i], cache.Xc[j]
    if issparse(Xi) and issparse(Xj):
        chr_cos = float(Xi.multiply(Xj).sum())

    emb_cos = 0.0
    if cache.embs is not None:
        vi, vj = cache.embs[i], cache.embs[j]
        emb_cos = float(np.clip(np.dot(vi, vj), -1.0, 1.0))

    hi, cols, hmax = helper_similarity(df.iloc[i].to_dict(), df.iloc[j].to_dict(), helper_cols)
    lr = (min(cache.len_arr[i], cache.len_arr[j]) / max(cache.len_arr[i], cache.len_arr[j])) if max(cache.len_arr[i], cache.len_arr[j]) else 0.0

    acr = acronym_similarity(
        ni, nj,
        cache.is_acronym[i], cache.is_acronym[j],
        cache.acr_clean[i], cache.acr_clean[j],
        cache.alias_map
    )

    return np.array([rf, chr_cos, emb_cos, jacc, overlap, hi, lr, acr], dtype=float)

def blend_score(feat: np.ndarray) -> float:
    # [rf, chr, emb, jacc, overlap, helper, len_ratio, acr]
    rf, chr_, emb, jacc, ov, h, lr, acr = feat
    w_rf, w_chr, w_emb, w_jac, w_ov, w_h, w_lr, w_acr = 0.33, 0.21, 0.10, 0.08, 0.02, 0.16, 0.02, 0.08
    synergy = 0.04 if (acr >= 0.99 and h >= 0.5) else 0.0
    s = (w_rf*rf + w_chr*chr_ + w_emb*emb + w_jac*jacc +
         w_ov*min(ov/5.0, 1.0) + w_h*h + w_lr*lr + w_acr*acr + synergy)
    return float(max(0.0, min(1.0, s)))

# --------------- Helper-driven acronym pairs ---------------
def helper_pairs_for_acronyms(cache: FeatureCache, strong_only: bool = True, bucket_cap:int=1000) -> List[Tuple[int,int]]:
    strong_prefixes = ("dom:", "phone:", "tax:", "vid:")
    pairs = set()
    for key, idxs in cache.helper_buckets.items():
        if strong_only and not key.startswith(strong_prefixes):
            continue
        if len(idxs) < 2 or len(idxs) > bucket_cap:
            continue
        has_acr = any(cache.is_acronym[i] for i in idxs)
        has_long = any(not cache.is_acronym[i] for i in idxs)
        if not (has_acr and has_long):
            continue
        acrs = [i for i in idxs if cache.is_acronym[i]]
        longs = [i for i in idxs if not cache.is_acronym[i]]
        for i in acrs:
            for j in longs:
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    return sorted(pairs)

# --------------- Threshold calibration ------------------
def sample_likely_negatives(df: pd.DataFrame, rid_col: str, helper_cols: List[str], max_pairs:int=20000) -> List[Tuple[int,int]]:
    idxs = list(range(len(df)))
    if len(idxs) < 2:
        return []
    country_col = None
    domain_col = None
    for c in helper_cols:
        cl = c.lower()
        if country_col is None and ("country" in cl or cl == "ctry" or "iso" in cl):
            country_col = c
        if domain_col is None and ("domain" in cl or "email" in cl or "website" in cl or "url" in cl):
            domain_col = c
    pairs = set()
    trials = 0
    while len(pairs) < max_pairs and trials < max_pairs*5:
        i, j = random.sample(idxs, 2)
        if i > j: i, j = j, i
        ok = False
        if country_col:
            ci, cj = str(df.iloc[i].get(country_col, "")), str(df.iloc[j].get(country_col, ""))
            if ci and cj and normalize_text(ci) != normalize_text(cj): ok = True
        if (not ok) and domain_col:
            di = etld1_from_value(str(df.iloc[i].get(domain_col,"")))
            dj = etld1_from_value(str(df.iloc[j].get(domain_col,"")))
            if di and dj and di != dj: ok = True
        if ok:
            pairs.add((i,j))
        trials += 1
    return list(pairs)

def calibrate_threshold_from_scores(scores_all: List[float], neg_scores: List[float]) -> List[float]:
    if not scores_all:
        return [0.70, 0.80, 0.90]
    if neg_scores:
        q97 = float(np.quantile(np.array(neg_scores), 0.97))
        t0 = max(0.55, min(0.88, q97))
    else:
        t0 = float(np.quantile(np.array(scores_all), 0.75))
        t0 = max(0.55, min(0.88, t0))
    t1 = min(0.95, t0 + 0.08)
    t2 = min(0.99, t0 + 0.15)
    thrs = sorted(set(round(x,3) for x in [t0,t1,t2] if 0.5 <= x <= 0.99))
    return thrs or [0.70, 0.80, 0.90]

# --------------- Build clusters + strong auto-merge ----------------
def build_clusters_from_pairs(
    idxs: List[int],
    pairs: List[Tuple[int,int,float]],
    threshold: float,
    must_groups_idx: List[List[int]],
    rid_list: List[str],
    cache: "FeatureCache",
    df: pd.DataFrame,
    helper_cols: List[str],
    auto_merge: bool = True
) -> List[List[int]]:
    """
    Build components at 'threshold', enforce approved must-link groups,
    then iteratively auto-merge components using helper+acronym+score rules.
    """
    # ---------- Base graph ----------
    idxs_set = set(idxs)
    adj = defaultdict(set)
    score_map: Dict[Tuple[int,int], float] = {}
    for i, j, s in pairs:
        a, b = (i, j) if i < j else (j, i)
        score_map[(a, b)] = max(score_map.get((a, b), 0.0), s)
        if s >= threshold and (i in idxs_set) and (j in idxs_set):
            adj[i].add(j); adj[j].add(i)

    # Must-link seeds: link all records inside each approved cluster (no split)
    for group in must_groups_idx:
        g = [i for i in group if i in idxs_set]
        for a in range(len(g)):
            for b in range(a+1, len(g)):
                ia, ib = g[a], g[b]
                adj[ia].add(ib); adj[ib].add(ia)

    # Connected components
    def connected_components() -> List[List[int]]:
        seen = set()
        comps: List[List[int]] = []
        for start in sorted(idxs_set):
            if start in seen: continue
            stack = [start]; seen.add(start); comp = [start]
            while stack:
                u = stack.pop()
                for v in adj.get(u, []):
                    if v not in seen:
                        seen.add(v); stack.append(v); comp.append(v)
            comps.append(sorted(comp))
        return comps

    comps = connected_components()
    if not auto_merge or len(comps) <= 1:
        return comps

    # ---------- Precompute per-component summaries ----------
    STRONG_PREFIXES = ("dom:", "phone:", "tax:", "vid:")
    WEAK_COUNTRY = "ctry:"
    SCORE_FOR_INIT_COUNTRY = 0.85  # high cross score for init+country merges

    def comp_helper_keys(comp: List[int]) -> Tuple[set, set]:
        strong, country = set(), set()
        for i in comp:
            for hk in cache.helper_keys[i]:
                if hk.startswith(STRONG_PREFIXES):
                    strong.add(hk)
                if hk.startswith(WEAK_COUNTRY):
                    country.add(hk)
        return strong, country

    def comp_initialisms(comp: List[int]) -> set:
        inits = set()
        for i in comp:
            if not cache.is_acronym[i]:
                init = initialism_from_norm(cache.norm[i])
                if len(init) >= 2:
                    inits.add(init)
        return inits

    def comp_acronyms(comp: List[int]) -> set:
        return {cache.acr_clean[i] for i in comp if cache.is_acronym[i] and cache.acr_clean[i]}

    def any_acronym_bridge(A: List[int], B: List[int]) -> bool:
        # Is there an acronym<->longform pair with some helper agreement?
        for i in A:
            for j in B:
                if cache.is_acronym[i] and (not cache.is_acronym[j]):
                    acr = is_acronym_of_clean(cache.acr_clean[i], cache.norm[j]) or \
                          (cache.norm[j] in cache.alias_map and cache.acr_clean[i] in cache.alias_map[cache.norm[j]])
                    if acr:
                        hi, _, hmax = helper_similarity(df.iloc[i].to_dict(), df.iloc[j].to_dict(), helper_cols)
                        if (hmax >= 0.5) or (hi >= 0.5):
                            return True
                if cache.is_acronym[j] and (not cache.is_acronym[i]):
                    acr = is_acronym_of_clean(cache.acr_clean[j], cache.norm[i]) or \
                          (cache.norm[i] in cache.alias_map and cache.acr_clean[j] in cache.alias_map[cache.norm[i]])
                    if acr:
                        hi, _, hmax = helper_similarity(df.iloc[i].to_dict(), df.iloc[j].to_dict(), helper_cols)
                        if (hmax >= 0.5) or (hi >= 0.5):
                            return True
        return False

    def max_cross_score(A: List[int], B: List[int]) -> float:
        best = 0.0
        for i in A:
            for j in B:
                a, b = (i, j) if i < j else (j, i)
                s = score_map.get((a, b), 0.0)
                if s > best:
                    best = s
        return best

    # ---------- Iterative merging ----------
    changed = True
    while changed:
        changed = False
        m = len(comps)
        if m <= 1: break

        comp_strong = []
        comp_country = []
        comp_inits = []
        comp_acrs = []
        for comp in comps:
            sset, cset = comp_helper_keys(comp)
            comp_strong.append(sset)
            comp_country.append(cset)
            comp_inits.append(comp_initialisms(comp))
            comp_acrs.append(comp_acronyms(comp))

        dsu = DSU(m)

        # Rule 1: share any STRONG helper key -> merge
        key2c = defaultdict(list)
        for cid, sset in enumerate(comp_strong):
            for k in sset:
                key2c[k].append(cid)
        for k, cids in key2c.items():
            base = cids[0]
            for cid in cids[1:]:
                dsu.union(base, cid)

        # Rule 2: acronym bridge + any helper hit between comps -> merge
        for x in range(m):
            for y in range(x+1, m):
                if dsu.find(x) == dsu.find(y): continue
                A, B = comps[x], comps[y]
                if any_acronym_bridge(A, B):
                    dsu.union(x, y)

        # Rule 3: same initialism + same country + high cross score -> merge
        for x in range(m):
            for y in range(x+1, m):
                if dsu.find(x) == dsu.find(y): continue
                if not (comp_inits[x] and comp_inits[y]): 
                    continue
                if not (comp_country[x] & comp_country[y]):
                    continue
                if comp_inits[x] & comp_inits[y]:
                    if max_cross_score(comps[x], comps[y]) >= SCORE_FOR_INIT_COUNTRY:
                        dsu.union(x, y)

        # Rule 4: small cluster absorption if shares STRONG with a larger one
        sizes = [len(c) for c in comps]
        for x in range(m):
            for y in range(x+1, m):
                rx, ry = dsu.find(x), dsu.find(y)
                if rx == ry: 
                    continue
                if not (sizes[x] <= 2 or sizes[y] <= 2):
                    continue
                if comp_strong[x] & comp_strong[y]:
                    dsu.union(x, y)

        # Rebuild comps from DSU
        new_groups = defaultdict(list)
        for cid in range(m):
            new_groups[dsu.find(cid)].extend(comps[cid])
        new_comps = [sorted(set(v)) for v in new_groups.values()]
        if len(new_comps) != len(comps) or any(set(a) != set(b) for a, b in zip(sorted(comps), sorted(new_comps))):
            comps = new_comps
            changed = True

    return comps

# --------------- IO: data loading & feedback ---------------
def load_all_csvs(input_dir: str, name_cols: List[str], helper_cols: List[str]) -> pd.DataFrame:
    """Load only CSVs that contain at least one requested NAME column (case-insensitive)."""
    req_names   = {canon_header(c) for c in name_cols}

    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    frames, skipped = [], []

    for p in paths:
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
        except Exception:
            df = pd.read_csv(p, low_memory=False)
        cols = {canon_header(c) for c in df.columns}
        has_name = bool(cols & req_names)
        if not has_name:
            skipped.append(os.path.basename(p))
            continue
        df["_source_file"] = os.path.basename(p)
        frames.append(df)

    if skipped:
        print(f"Skipping {len(skipped)} file(s) with no requested name columns: {skipped}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def pick_name_value(row: pd.Series, name_cols: List[str]) -> str:
    for c in name_cols:
        if c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return ""

def ensure_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = ""

def read_feedback_approvals(feedback_dir: str, round_num:int) -> Dict[str,str]:
    path = os.path.join(feedback_dir, f"approvals_round_{round_num:02d}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        adf = pd.read_csv(path, dtype=str)
        for _, r in adf.iterrows():
            cid = str(r.get("cluster_id","")).strip()
            dec = str(r.get("decision","")).strip().lower()
            if cid:
                out[cid] = dec
    except Exception:
        pass
    return out

def write_proposals(feedback_dir: str, round_num:int, review_clusters: List[List[int]],
                    df: pd.DataFrame, rid_col:str, name_col:str,
                    scores_by_pair: Dict[Tuple[int,int], float]):
    # 1) Member-level proposals (rows only for clusters needing review)
    rows = []
    for k, comp in enumerate(review_clusters, 1):
        # cohesion (avg pair score inside comp, if available)
        if len(comp) <= 1:
            coh = 1.0
        else:
            acc = 0.0; cnt = 0
            for i in range(len(comp)):
                for j in range(i+1, len(comp)):
                    a, b = comp[i], comp[j]
                    key = (a,b) if a<b else (b,a)
                    if key in scores_by_pair:
                        acc += scores_by_pair[key]; cnt += 1
            coh = (acc / cnt) if cnt else 0.0

        cluster_id = f"R{round_num}_C{k}"
        for idx in comp:
            rows.append({
                "cluster_id": cluster_id,
                "rid": str(df.iloc[idx][rid_col]),
                "name": df.iloc[idx][name_col],
                "source": df.iloc[idx]["_source_file"],
                "cohesion_estimate": round(coh, 3)
            })

    out = pd.DataFrame(rows)
    prop_path = os.path.join(feedback_dir, f"proposals_round_{round_num:02d}.csv")
    out.to_csv(prop_path, index=False)

    # 2) Approvals file (one row per review cluster) with "items"
    ap_path = os.path.join(feedback_dir, f"approvals_round_{round_num:02d}.csv")

    def _item_label(rid_val, name_val):
        if is_pure_acronym_raw(name_val):
            return f"{rid_val} — {name_val}"
        init = initialism_from_norm(normalize_text(name_val))
        if init and clean_acronym(name_val) != init:
            return f"{rid_val} — {name_val} [{init}]"
        return f"{rid_val} — {name_val}"

    ap_rows = []
    for cid, sub in out.groupby("cluster_id", sort=True):
        size = int(len(sub))
        coh = float(sub["cohesion_estimate"].iloc[0]) if "cohesion_estimate" in sub else 0.0
        items_str = " | ".join(_item_label(r, n) for r, n in zip(sub["rid"], sub["name"]))
        ap_rows.append({
            "cluster_id": cid,
            "size": size,
            "cohesion": round(coh, 3),
            "items": items_str,
            "decision": ""  # preserve below if the file already exists
        })
    ap_new = pd.DataFrame(ap_rows, columns=["cluster_id","size","cohesion","items","decision"])

    if os.path.exists(ap_path):
        try:
            ap_old = pd.read_csv(ap_path, dtype=str)
            if "cluster_id" in ap_old.columns:
                old_dec = dict(zip(ap_old["cluster_id"].astype(str),
                                   ap_old["decision"] if "decision" in ap_old.columns else [""]*len(ap_old)))
                ap_new["decision"] = ap_new["cluster_id"].map(old_dec).fillna("")
        except Exception:
            pass

    ap_new.to_csv(ap_path, index=False)

# --------------- Main pipeline -----------------
def main():
    ap = argparse.ArgumentParser(description="Entity equivalence clustering with human approvals only (auto-merge enabled)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--feedback_dir", required=True)
    ap.add_argument("--name_cols", required=True, help="Comma-separated columns that define the entity string (first non-empty wins)")
    ap.add_argument("--helper_cols", default="", help="Comma-separated helper columns (e.g., country,zip,domain,email,website,phone,hs_code,vendor_id,tax_id)")
    ap.add_argument("--id_col", default="", help="Optional unique id column; if absent, a synthetic rid is created")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feedback_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "state.json")
    st = load_state(state_path)

    # Parse column params early (for loader)
    name_cols = [c.strip() for c in args.name_cols.split(",") if c.strip()]
    helper_cols = [c.strip() for c in args.helper_cols.split(",") if c.strip()]

    # Load data (skip files that lack any requested name column)
    df = load_all_csvs(args.input_dir, name_cols, helper_cols)
    if df.empty:
        print("No usable CSVs found (none contained the requested name columns).")
        save_state(state_path, st)
        sys.exit(0)

    # Ensure merged frame has all needed columns
    ensure_columns(df, name_cols + helper_cols)

    # rid column
    rid_col = args.id_col.strip() if args.id_col.strip() else "_rid"
    if rid_col not in df.columns:
        df[rid_col] = [f"row_{i}" for i in range(len(df))]

    # Build entity name (first non-empty from name_cols)
    df["_name_raw"] = df.apply(lambda r: pick_name_value(r, name_cols), axis=1)
    df["_name_norm"] = df["_name_raw"].apply(normalize_text)

    # Filter out empty names
    keep_mask = df["_name_raw"].astype(str).str.strip().astype(bool)
    df = df[keep_mask].reset_index(drop=True)

    # Apply approvals from previous proposals of the CURRENT round (if provided)
    approvals = read_feedback_approvals(args.feedback_dir, st.round)
    if approvals:
        prop_path = os.path.join(args.feedback_dir, f"proposals_round_{st.round:02d}.csv")
        if os.path.exists(prop_path):
            prop = pd.read_csv(prop_path, dtype=str)
            for cid, decision in approvals.items():
                if decision in {"approve","approved","yes","y"}:
                    members = prop.loc[prop["cluster_id"]==cid, "rid"].tolist()
                    if members:
                        st.approved_clusters.append(members)
        # advance to next round after applying approvals
        st.round += 1

    # Build feature cache over ALL rows
    cache = FeatureCache(df["_name_raw"].tolist(), df, helper_cols)

    # Candidate pair generation
    slices = generate_blocked_slices(cache, cap=5000)
    active_idx = list(range(len(df)))  # everyone stays active; approved groups become must-links

    # Candidate pairs & scores
    pairs_global: Dict[Tuple[int,int], float] = {}
    active_set = set(active_idx)
    for sl in slices:
        sl = [i for i in sl if i in active_set]
        if len(sl) < 2:
            continue
        n = len(sl)
        k = min(max(10, int(math.sqrt(n))+6), n)
        Xc_sub = cache.Xc[sl]
        tfidf_pairs = knn_pairs_sparse(Xc_sub, k)
        emb_pairs = []
        if cache.embs is not None:
            emb_pairs = knn_pairs_dense(cache.embs[sl], k)

        cand = set()
        for i,j,_ in tfidf_pairs:
            gi, gj = sl[i], sl[j]
            if gi > gj: gi, gj = gj, gi
            cand.add((gi,gj))
        for i,j,_ in emb_pairs:
            gi, gj = sl[i], sl[j]
            if gi > gj: gi, gj = gj, gi
            cand.add((gi,gj))

        for (i,j) in cand:
            feat = pair_features(i, j, cache, df, helper_cols)
            sc = blend_score(feat)
            key = (i,j)
            if key in pairs_global:
                pairs_global[key] = max(pairs_global[key], sc)
            else:
                pairs_global[key] = sc

    # Ensure acronym<->longform pairs get considered via helper buckets
    for (i,j) in helper_pairs_for_acronyms(cache, strong_only=True, bucket_cap=1200):
        key = (i,j)
        feat = pair_features(i, j, cache, df, helper_cols)
        sc = blend_score(feat)
        if key in pairs_global:
            pairs_global[key] = max(pairs_global[key], sc)
        else:
            pairs_global[key] = sc

    # Thresholds
    scores_all = [s for (i,j), s in pairs_global.items() if (i in active_set and j in active_set)]
    neg_pairs = sample_likely_negatives(df, rid_col, helper_cols, max_pairs=min(20000, len(scores_all)*2 or 2000))
    neg_scores = []
    for (i,j) in neg_pairs:
        key = (i,j) if i<j else (j,i)
        if key in pairs_global:
            neg_scores.append(pairs_global[key])
        else:
            feat = pair_features(i, j, cache, df, helper_cols)
            neg_scores.append(blend_score(feat))

    thresholds = calibrate_threshold_from_scores(scores_all, neg_scores)
    st.thresholds = thresholds

    scored_pairs = [(i,j,s) for (i,j), s in pairs_global.items()]

    # Build must-link groups from previously approved clusters (by index)
    rid_to_idx = {str(df.iloc[i][rid_col]): i for i in range(len(df))}
    must_groups_idx: List[List[int]] = []
    for grp in st.approved_clusters:
        idxs = [rid_to_idx[r] for r in grp if r in rid_to_idx]
        if len(idxs) >= 2:
            must_groups_idx.append(sorted(set(idxs)))

    # Build clusters at current round threshold + strong auto-merge
    thr = thresholds[min(st.round-1, len(thresholds)-1)]
    clusters_idx = build_clusters_from_pairs(
        idxs=active_idx,
        pairs=scored_pairs,
        threshold=thr,
        must_groups_idx=must_groups_idx,
        rid_list=df[rid_col].astype(str).tolist(),
        cache=cache,
        df=df,
        helper_cols=helper_cols,
        auto_merge=True
    )

    # If nothing proposed, try next stricter threshold in this run to avoid “empty” round
    tried = 0
    while not clusters_idx and tried < len(thresholds)-1:
        tried += 1
        thr_next = thresholds[min(st.round-1+tried, len(thresholds)-1)]
        clusters_idx = build_clusters_from_pairs(
            idxs=active_idx,
            pairs=scored_pairs,
            threshold=thr_next,
            must_groups_idx=must_groups_idx,
            rid_list=df[rid_col].astype(str).tolist(),
            cache=cache,
            df=df,
            helper_cols=helper_cols,
            auto_merge=True
        )

    # Prepare mapping + decide which clusters need review
    rid_list = df[rid_col].astype(str).tolist()
    comp_rids = [sorted(rid_list[i] for i in comp) for comp in clusters_idx]
    comp_ridsets = [frozenset(cr) for cr in comp_rids]

    approved_sets = [frozenset(grp) for grp in st.approved_clusters]
    approved_setset = set(approved_sets)

    rid_to_cluster = {}
    review_clusters = []
    approved_index = {frozenset(grp): k+1 for k, grp in enumerate(st.approved_clusters)}

    review_counter = 0
    for comp, comp_set, comp_rid_list in zip(clusters_idx, comp_ridsets, comp_rids):
        if comp_set in approved_setset:
            fzk = approved_index.get(comp_set, len(approved_index)+1)
            approved_index[comp_set] = fzk
            for rid in comp_rid_list:
                rid_to_cluster[rid] = f"FZ{fzk}"
        else:
            review_counter += 1
            for rid in comp_rid_list:
                rid_to_cluster[rid] = f"R{st.round}_C{review_counter}"
            review_clusters.append(comp)

    mapping = pd.DataFrame({
        rid_col: df[rid_col].astype(str),
        "name": df["_name_raw"],
        "cluster_id": [rid_to_cluster.get(str(df.iloc[i][rid_col]), "") for i in range(len(df))]
    })
    out_map = os.path.join(args.output_dir, "clusters_latest.csv")
    mapping.to_csv(out_map, index=False)

    # Write proposals JUST for clusters needing review
    scores_by_pair = {k:v for k,v in pairs_global.items()}
    write_proposals(args.feedback_dir, st.round, review_clusters, df, rid_col, "_name_raw", scores_by_pair)

    save_state(state_path, st)

    print(f"Round {st.round} proposals written to {os.path.join(args.feedback_dir, f'proposals_round_{st.round:02d}.csv')}")
    print(f"Latest cluster mapping written to {out_map}")
    print("Next step:")
    print(f" - Open approvals_round_{st.round:02d}.csv and mark which clusters to approve (decision=approve).")
    print(" - Re-run this script to refine with your feedback. Approved clusters won’t split, and can merge/gain members automatically if evidence is strong.")

if __name__ == "__main__":
    main()

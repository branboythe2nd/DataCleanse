#!/usr/bin/env python3
"""
Entity-Equivalence Clusterer — Dynamic, Precise, and Self-Updating (TXT per CSV)

What this script does
---------------------
1) Normalizes names (lowercase, accents, '&'/'+' -> 'and') and learns dataset-specific stopwords.
2) Builds candidate pairs via:
   - semantic kNN (all-mpnet-base-v2 sentence embeddings, unit vectors),
   - char 3–5gram TF-IDF kNN,
   - rare-token pairs (rare, non-generic tokens, length>=4),
   - acronym bridges (e.g., "VPI&SU" -> "VPISU" against full expansion).
3) Featurizes each candidate pair (semantic cosine, char cosine, token Jaccard/overlap
   sans generics, length/sequence similarities, acronym flags, etc.).
4) Trains/updates a persistent verifier (SGDClassifier with partial_fit) from high-confidence
   seeds; optionally ingests your labeled feedback CSV.
5) Keeps edges scoring >= a calibrated threshold AND passing a structure gate:
   share a specific (non-generic) token or be an explicit acronym bridge.
6) Clusters via connected components, purifies by centroid + token-core overlap,
   and runs a strict acronym post-pass (ambiguous acronyms -> "misc").
7) Writes one TXT per CSV with clusters and frequencies. Optionally exports a
   review CSV of borderline pairs for you to label next time.

Required CLI args
-----------------
--input-dir   folder with CSV files
--output-dir  folder to write TXT outputs
--columns     comma-separated list of candidate column names (first present is used)

Optional
--------
--feedback      CSV of human-labeled pairs (columns: left,right,label)
--model-dir     persistent verifier dir (default: ./model_store/entity_verifier)
--emit-review   number of borderline pairs to export (default: 0 = off)
--review-dir    folder for review CSVs (default: same as --output-dir). The CSVs
                include an empty 'label' column for you to fill (1=same, 0=different).

Example
-------
pip install pandas numpy scipy scikit-learn sentence-transformers

python entity_cluster_persistent.py \
  --input-dir ./data \
  --output-dir ./clusters_out \
  --columns name,company,shipper \
  --emit-review 150 \
  --review-dir ./to_label
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ----------------------------- Config & constants ----------------------------- #

MISC_LABEL = -999_999

# Generic tokens rarely identify an entity on their own (domain-agnostic base)
GENERIC_ALWAYS = {
    "university","college","institute","school","polytechnic","technology","tech",
    "state","company","co","inc","corp","corporation","ltd","llc","group",
    "solutions","systems","international","department","division","services",
}

PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
MULTISPACE_RE = re.compile(r"\s+")
ONLY_ALPHA_RE = re.compile(r"^[a-z]+$")

# ----------------------------- Text & token helpers --------------------------- #

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def _preclean(s: str) -> str:
    # Normalize symbols early so both tokens and acronyms see the same string
    return s.replace("&", " and ").replace("+", " and ")

def tokenize(name: str) -> List[str]:
    s = strip_accents(str(name).lower().strip())
    s = _preclean(s)
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return [t for t in s.split() if t]

def alias_fold(tokens: List[str]) -> List[str]:
    """
    Domain-agnostic, conservative alias fold:
      - If both 'polytechnic' and 'institute' appear, also add synthetic 'tech'
        to better align with 'Tech' aliases (e.g., Virginia Tech).
    """
    toks = list(tokens)
    ts = set(toks)
    if "polytechnic" in ts and "institute" in ts and "tech" not in ts:
        toks.append("tech")
    return toks

def dynamic_stopwords(names: List[str], top_frac: float = 0.30) -> Set[str]:
    # tokens appearing in > top_frac of names + short tokens + connectors
    df = Counter()
    seen = 0
    for s in names:
        seen += 1
        df.update(set(tokenize(s)))
    hi = {t for t, c in df.items() if c >= top_frac * max(1, seen)}
    short = {t for t in df if len(t) <= 2}
    base_connectors = {"of","the","and","at","for","in","&","de","del","la","el","los","las"}
    return hi | short | base_connectors

def informative_tokens(name: str, dyn_stop: Set[str]) -> Set[str]:
    toks = alias_fold(tokenize(name))
    return {t for t in toks if t not in dyn_stop and len(t) >= 3}

def build_generic_bucket(names: List[str], dyn_stop: Set[str], frac: float = 0.15) -> Set[str]:
    """
    Data-driven + universal list:
      - tokens that appear in >= frac of names (dataset-generic)
      - plus a curated generic set
      - plus dynamic stopwords
    """
    df = Counter()
    for s in names:
        df.update(set(tokenize(s)))
    n = max(1, len(names))
    dataset_generic = {t for t, c in df.items() if (c / n) >= frac}
    return dataset_generic | GENERIC_ALWAYS | set(dyn_stop)

def norm_for_embedding(name: str) -> str:
    toks = alias_fold(tokenize(name))
    return " ".join(toks) if toks else strip_accents(str(name).lower())

def tokens_for_acronym(name: str, dyn_stop: Set[str]) -> List[str]:
    toks = alias_fold(tokenize(name))
    return [t for t in toks if t not in dyn_stop]

def acronym_from_tokens(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens if t)

def is_acronym_only(name: str) -> bool:
    s = strip_accents(_preclean(str(name).lower()))
    s = PUNCT_RE.sub("", s)
    return 2 <= len(s) <= 12 and ONLY_ALPHA_RE.match(s) is not None

def acronym_key_for_acronym_only(name: str) -> str:
    """
    Build a normalized acronym key for an acronym-only string.
    Example: "VPI&SU" -> "VPISU" (drop non-alpha and the substring "and").
    """
    raw = strip_accents(_preclean(name.lower()))
    key = re.sub(r"[^a-z]", "", raw)
    key = key.replace("and", "")
    return key.upper()

# ----------------------------- I/O & counts ----------------------------------- #

def pick_column_for_file(csv_path: Path, candidates: List[str]) -> Optional[str]:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return None
    cols = set(map(str, header.columns))
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def load_counts(csv_path: Path, column: str, chunksize: int = 200_000) -> Tuple[List[str], Dict[str, Counter]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for chunk in pd.read_csv(csv_path, usecols=[column], chunksize=chunksize, dtype={column: str}):
        s = chunk[column].fillna("").astype(str)
        for name in s:
            counts[name][name] += 1
    unique_names = list(counts.keys())
    return unique_names, counts

# ----------------------------- Embeddings & vectors --------------------------- #

def embed_all(names: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode([norm_for_embedding(n) for n in names], normalize_embeddings=True, show_progress_bar=False)

def char_tfidf(names: List[str]) -> csr_matrix:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), lowercase=True, min_df=1, norm="l2")
    return vec.fit_transform([strip_accents(_preclean(n.lower())) for n in names])

def knn_pairs_from_vectors(X, k: int) -> List[Tuple[int,int,float]]:
    n = X.shape[0]
    if n <= 1:
        return []
    k = max(2, min(k, n))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    pairs = []
    for i in range(n):
        for r in range(1, k):  # skip self
            j = idxs[i, r]
            if i < j:
                sim = 1.0 - float(dists[i, r])
                pairs.append((i, j, sim))
    return pairs

# ----------------------------- Candidate generation --------------------------- #

def build_candidate_pairs(names: List[str],
                          embs: np.ndarray,
                          Xc: csr_matrix,
                          dyn_stop: Set[str],
                          generic_bucket: Set[str]) -> Tuple[
                              List[Tuple[int,int,float,float]],
                              List[Set[str]],
                              Set[Tuple[int,int]]
                          ]:
    """
    Return:
      - pairs: list of (i,j, semantic_cos, char_cos)
      - toks_per: List[Set[str]] informative tokens per name
      - acr_pairs: set of pairs created by explicit acronym bridging
    """
    n = len(names)
    k = min(max(10, int(np.sqrt(n)) + 3), n)

    emb_pairs = knn_pairs_from_vectors(embs, k=k)
    chr_pairs = knn_pairs_from_vectors(Xc,  k=k)

    cand: Dict[Tuple[int,int], List[Optional[float]]] = {}
    def add_pair(i, j, s_cos=None, s_chr=None):
        if i > j: i, j = j, i
        if (i,j) not in cand:
            cand[(i,j)] = [s_cos, s_chr]
        else:
            cur = cand[(i,j)]
            cand[(i,j)] = [
                max(cur[0], s_cos) if (cur[0] is not None and s_cos is not None) else (cur[0] if cur[0] is not None else s_cos),
                max(cur[1], s_chr) if (cur[1] is not None and s_chr is not None) else (cur[1] if cur[1] is not None else s_chr),
            ]

    for i, j, s in emb_pairs: add_pair(i, j, s_cos=s, s_chr=None)
    for i, j, s in chr_pairs: add_pair(i, j, s_cos=None, s_chr=s)

    # Rare-token booster (very rare, non-generic, long enough)
    token_df = Counter()
    toks_per: List[Set[str]] = []
    for s in names:
        ts = informative_tokens(s, dyn_stop)
        toks_per.append(ts)
        token_df.update(ts)

    very_rare_thr = max(2, int(np.ceil(0.05 * n)))  # <= 5% of names
    bucket: Dict[str, List[int]] = defaultdict(list)
    for i, ts in enumerate(toks_per):
        for t in ts:
            if (len(t) >= 4) and (token_df[t] <= very_rare_thr) and (t not in generic_bucket):
                bucket[t].append(i)
    for t, idxs in bucket.items():
        if len(idxs) < 2: continue
        idxs = sorted(idxs)
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                add_pair(idxs[a], idxs[b], s_cos=None, s_chr=0.0)  # verifier will decide

    # Acronym-bridging candidates
    acr_only_idx = [i for i, n in enumerate(names) if is_acronym_only(n)]
    exp_idx = [i for i in range(n) if i not in acr_only_idx]
    exp_acr: Dict[str, List[int]] = defaultdict(list)
    for i in exp_idx:
        acr = acronym_from_tokens(tokens_for_acronym(names[i], dyn_stop))
        acr = acr.replace("and", "")  # defensively drop "and" if present
        if 2 <= len(acr) <= 12:
            exp_acr[acr.upper()].append(i)
    acr_pairs: Set[Tuple[int,int]] = set()
    for i in acr_only_idx:
        key = acronym_key_for_acronym_only(names[i])
        if key in exp_acr:
            for j in exp_acr[key]:
                a, b = (i, j) if i < j else (j, i)
                acr_pairs.add((a, b))
                add_pair(a, b, s_cos=None, s_chr=0.0)

    pairs = sorted([(i, j, cand[(i,j)][0] or 0.0, cand[(i,j)][1] or 0.0) for (i,j) in cand])
    return pairs, toks_per, acr_pairs

# ----------------------------- Pairwise features ----------------------------- #

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def specific_tokens_from_cache(toks_per: List[Set[str]], generic_bucket: Set[str], i: int) -> Set[str]:
    return {t for t in toks_per[i] if t not in generic_bucket}

def pair_features(names: List[str],
                  embs: np.ndarray,
                  Xc: csr_matrix,
                  pairs: List[Tuple[int,int,float,float]],
                  dyn_stop: Set[str],
                  generic_bucket: Set[str],
                  toks_per: List[Set[str]],
                  acr_pairs: Set[Tuple[int,int]]) -> np.ndarray:
    """
    Feature vector per pair (fixed dimension -> persisted model remains valid):
      0 semantic cosine
      1 char cosine
      2 Jaccard of informative tokens (no generics)
      3 specific-token overlap count
      4 exact same acronym (non-empty)
      5 one acronym equals other's initials
      6 length ratio (min/ max of normalized strings)
      7 ordered informative-token sequence similarity
      8 raw normalized string sequence similarity
      9 heuristic: (overlap>=1 and (cos>=.60 or char>=.60))
     10 is_explicit_acronym_bridge (from candidate builder)
    """
    nfeat = 11
    feats = np.zeros((len(pairs), nfeat), dtype=float)

    acrs = []
    lens = []
    normed = []
    for s in names:
        acr = acronym_from_tokens(tokens_for_acronym(s, dyn_stop)).upper()
        acr = acr.replace("AND", "")  # normalize
        acrs.append(acr)
        lens.append(len(strip_accents(_preclean(s.lower()))))
        normed.append(norm_for_embedding(s))

    for k, (i, j, s_cos, s_chr) in enumerate(pairs):
        ti = specific_tokens_from_cache(toks_per, generic_bucket, i)
        tj = specific_tokens_from_cache(toks_per, generic_bucket, j)
        ji = jaccard(ti, tj)
        ov = len(ti & tj)

        ai, aj = acrs[i], acrs[j]
        acr_equal_nonempty = int(bool(ai) and bool(aj) and (ai == aj))

        target_j = re.sub(r"[^A-Z]", "", names[j].upper())
        target_i = re.sub(r"[^A-Z]", "", names[i].upper())
        acr_one_exact_of_other = int((bool(ai) and (ai == target_j)) or (bool(aj) and (aj == target_i)))

        lr = min(lens[i], lens[j]) / max(lens[i], lens[j]) if max(lens[i], lens[j]) else 0.0

        ti_str = " ".join(sorted(list(ti)))
        tj_str = " ".join(sorted(list(tj)))
        tok_seq = seq_ratio(ti_str, tj_str)

        raw_seq = seq_ratio(normed[i], normed[j])

        is_acr_bridge = int(((i, j) in acr_pairs) or ((j, i) in acr_pairs))

        feats[k, :] = [
            s_cos, s_chr, ji, ov, acr_equal_nonempty, acr_one_exact_of_other,
            lr, tok_seq, raw_seq,
            1.0 if (ov >= 1 and (s_cos >= 0.60 or s_chr >= 0.60)) else 0.0,
            is_acr_bridge
        ]
    return feats

# ----------------------------- Persistent verifier --------------------------- #

@dataclass
class VerifierMeta:
    version: int = 1
    n_updates: int = 0
    n_seen_pairs: int = 0
    threshold_: float = 0.5
    t_low_: float = 0.45
    t_high_: float = 0.55
    n_features_: int = 11  # feature dimension check

class PersistentVerifier:
    """
    Online pairwise verifier that persists across runs.
    - Features in, probability out (same-entity?).
    - partial_fit enables incremental learning.
    - Stores scaler, classifier, and threshold in model_dir.
    """
    def __init__(self, model_dir: str, n_features: int):
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[SGDClassifier] = None
        self.meta = VerifierMeta(n_features_=n_features)
        self._classes = np.array([0, 1], dtype=int)
        self._load_or_init(n_features)

    def _load_or_init(self, n_features: int):
        sc_path = self.dir / "scaler.joblib"
        clf_path = self.dir / "clf.joblib"
        meta_path = self.dir / "meta.json"
        if sc_path.exists() and clf_path.exists() and meta_path.exists():
            self.scaler = joblib.load(sc_path)
            self.clf = joblib.load(clf_path)
            self.meta = VerifierMeta(**json.loads(meta_path.read_text()))
            # If feature dimension changed across versions, reset cleanly
            if self.meta.n_features_ != n_features:
                self.scaler = StandardScaler(with_mean=False)
                self.clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000)
                self.meta = VerifierMeta(n_features_=n_features)
                self._save_all()
        else:
            self.scaler = StandardScaler(with_mean=False)
            self.clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000)
            self.meta = VerifierMeta(n_features_=n_features)
            self._save_all()

    def _save_all(self):
        joblib.dump(self.scaler, self.dir / "scaler.joblib")
        joblib.dump(self.clf,    self.dir / "clf.joblib")
        (self.dir / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))

    def _calibrate_threshold(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        fpr, tpr, thr = roc_curve(y_true, probs)
        j = tpr - fpr
        t = float(max(0.5, thr[np.argmax(j)]))  # precision-first: never below 0.5
        self.meta.threshold_ = t
        self.meta.t_low_ = max(0.0, t - 0.05)
        self.meta.t_high_ = min(1.0, t + 0.05)
        return t

    def bootstrap_or_update(self, X_seeds: np.ndarray, y_seeds: np.ndarray):
        # Update scaler
        self.scaler.partial_fit(X_seeds)
        Xs = self.scaler.transform(X_seeds)

        # Class balance via sample weights
        cls, counts = np.unique(y_seeds, return_counts=True)
        weights = {int(c): (counts.sum() / (len(cls) * cnt)) for c, cnt in zip(cls, counts)}
        sw = np.array([weights[int(y)] for y in y_seeds], dtype=float)

        # Initial call must include 'classes'
        self.clf.partial_fit(Xs, y_seeds, classes=self._classes, sample_weight=sw)

        # Calibrate threshold on seeds
        probs = self.clf.predict_proba(Xs)[:, 1]
        self._calibrate_threshold(probs, y_seeds)

        self.meta.n_updates += 1
        self.meta.n_seen_pairs += int(X_seeds.shape[0])
        self._save_all()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    def threshold(self) -> float:
        return self.meta.threshold_

# ----------------------------- Seeds & feedback ------------------------------ #

def build_seed_labels(names: List[str],
                      pairs: List[Tuple[int,int,float,float]],
                      feats: np.ndarray,
                      dyn_stop: Set[str],
                      min_seed: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic seeds for self-training:
      Pos: exact initialism match, acronym equals other's initials, or strong lexical evidence.
      Neg: zero specific-token overlap and low sims.
    Ensures at least ~min_seed/2 of each; relaxes as needed.
    """
    idxs = np.arange(len(pairs))
    pos_idx, neg_idx = [], []

    for k, (i, j, s_cos, s_chr) in enumerate(pairs):
        ji = feats[k, 2]
        ov = feats[k, 3]
        acr_equal = bool(feats[k, 4])
        acr_initial = bool(feats[k, 5])
        strong_lex = (ov >= 2 and s_chr >= 0.80) or (ji >= 0.85)
        if acr_equal or acr_initial or strong_lex:
            pos_idx.append(k)

    for k, (i, j, s_cos, s_chr) in enumerate(pairs):
        ov = feats[k, 3]
        if ov == 0 and s_cos <= 0.35 and s_chr <= 0.35:
            neg_idx.append(k)

    # Relax if needed
    if len(pos_idx) < min_seed // 2:
        for k, (i, j, s_cos, s_chr) in enumerate(pairs):
            if k in pos_idx: continue
            ji = feats[k, 2]; ov = feats[k, 3]
            if (ov >= 1 and (s_cos >= 0.65 or s_chr >= 0.65)) or ji >= 0.75:
                pos_idx.append(k)
            if len(pos_idx) >= min_seed // 2: break

    if len(neg_idx) < min_seed // 2:
        for k, (i, j, s_cos, s_chr) in enumerate(pairs):
            if k in neg_idx: continue
            if feats[k, 3] == 0 and feats[k, 0] <= 0.45 and feats[k, 1] <= 0.45:
                neg_idx.append(k)
            if len(neg_idx) >= min_seed // 2: break

    sel = sorted(set(pos_idx[:min_seed]) | set(neg_idx[:min_seed]))
    y = np.zeros(len(sel), dtype=int)
    for t, k in enumerate(sel):
        y[t] = 1 if k in pos_idx else 0
    return y, np.array(sel, dtype=int)

def load_feedback(feedback_csv: Path,
                  names: List[str],
                  embs: np.ndarray,
                  Xc: csr_matrix,
                  dyn_stop: Set[str],
                  generic_bucket: Set[str],
                  toks_per: List[Set[str]],
                  acr_pairs: Set[Tuple[int,int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feedback CSV schema: left,right,label
    Returns features and labels for rows where both names are found.
    """
    df = pd.read_csv(feedback_csv)
    need_cols = {"left","right","label"}
    cols_lower = {c.lower(): c for c in df.columns}
    if not need_cols.issubset(cols_lower.keys()):
        return np.empty((0,11)), np.empty((0,), dtype=int)

    left_col, right_col, label_col = cols_lower["left"], cols_lower["right"], cols_lower["label"]

    index = {n: i for i, n in enumerate(names)}
    rows = []
    pairs = []
    for _, r in df.iterrows():
        l, rname, lab = str(r[left_col]), str(r[right_col]), int(r[label_col])
        if l in index and rname in index:
            i, j = index[l], index[rname]
            if i > j: i, j = j, i
            pairs.append((i, j, 0.0, 0.0))  # sims recalculated in features
            rows.append(lab)
    if not pairs:
        return np.empty((0,11)), np.empty((0,), dtype=int)

    feats = pair_features(names, embs, Xc, pairs, dyn_stop, generic_bucket, toks_per, acr_pairs)
    y = np.array(rows, dtype=int)
    return feats, y

# ----------------------------- Graph & clustering ---------------------------- #

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

def compute_centroids(labels: List[int], embs: np.ndarray) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    agg: Dict[int, List[np.ndarray]] = defaultdict(list)
    for lab, v in zip(labels, embs):
        if lab >= 0:
            agg[lab].append(v)
    for lab, vs in agg.items():
        cents[lab] = np.mean(np.vstack(vs), axis=0)
    return cents

def purify_clusters(names: List[str], labels: List[int], embs: np.ndarray,
                    dyn_stop: Set[str], generic_bucket: Set[str]) -> List[int]:
    cents = compute_centroids(labels, embs)
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab >= 0:
            cluster_members[lab].append(i)

    # Precompute tokens once
    toks_all = [informative_tokens(n, dyn_stop) for n in names]

    # Cluster core = specific tokens seen in ≥2 members
    core_tokens: Dict[int, Set[str]] = {}
    for lab, idxs in cluster_members.items():
        freq = Counter()
        for i in idxs:
            specific = {t for t in toks_all[i] if t not in generic_bucket}
            freq.update(specific)
        core_tokens[lab] = {t for t, c in freq.items() if c >= 2}

    sims_all = []
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab)
        if cen is None: continue
        for i in idxs:
            sims_all.append(cosine_sim(embs[i], cen))
    sim_gate = max(0.45, float(np.percentile(sims_all, 20))) if sims_all else 0.5

    new_labels = labels[:]
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab)
        core = core_tokens.get(lab, set())
        for i in idxs:
            specific_i = {t for t in toks_all[i] if t not in generic_bucket}
            has_overlap = len(specific_i & core) > 0
            sim_ok = (cen is not None) and (cosine_sim(embs[i], cen) >= sim_gate)
            if not (has_overlap and sim_ok):
                new_labels[i] = -1
    return new_labels

def post_assign_acronyms_strict(names: List[str],
                                labels: List[int],
                                counts: Dict[str, Counter],
                                embs: np.ndarray,
                                dyn_stop: Set[str]) -> Tuple[List[int], Counter]:
    votes: Dict[str, Counter] = defaultdict(Counter)
    acr_to_clusters: Dict[str, Set[int]] = defaultdict(set)
    exp_for_acr: Dict[str, List[int]] = defaultdict(list)

    for idx, (lab, name) in enumerate(zip(labels, names)):
        if lab < 0: continue
        acr = acronym_from_tokens(tokens_for_acronym(name, dyn_stop)).upper()
        acr = acr.replace("AND", "")
        if 2 <= len(acr) <= 12:
            votes[acr][lab] += 1
            acr_to_clusters[acr].add(lab)
            exp_for_acr[acr].append(idx)

    cents = compute_centroids(labels, embs)
    sims_all = []
    for i, lab in enumerate(labels):
        if lab < 0: continue
        cen = cents.get(lab)
        if cen is None: continue
        sims_all.append(cosine_sim(embs[i], cen))
    acr_sim_gate = max(0.50, float(np.percentile(sims_all, 30))) if sims_all else 0.55

    acr_indices = [i for i, n in enumerate(names) if is_acronym_only(n)]
    misc = Counter()
    for i in acr_indices:
        key = acronym_key_for_acronym_only(names[i])
        cset = acr_to_clusters.get(key, set())
        if len(cset) == 1:
            cid = next(iter(cset))
            votes_ok = votes[key][cid] >= 2
            cen = cents.get(cid, None)
            sim_ok = (cen is not None) and (cosine_sim(embs[i], cen) >= acr_sim_gate)
            if votes_ok or sim_ok:
                labels[i] = cid
                continue
        if len(cset) > 1 or votes.get(key, Counter()).total() > 0:
            labels[i] = MISC_LABEL
            misc.update(counts.get(names[i], Counter()))
        # else: no expansions -> remain singleton
    return labels, misc

# ----------------------------- Output writers -------------------------------- #

def write_txt(out_path: Path,
              labels: List[int],
              names: List[str],
              counts: Dict[str, Counter],
              misc_counter: Optional[Counter] = None) -> None:
    # Positive clusters
    cluster_to_counter: Dict[int, Counter] = defaultdict(Counter)
    for lab, name in zip(labels, names):
        if lab >= 0:
            cluster_to_counter[lab].update(counts.get(name, Counter()))

    ordered = sorted(cluster_to_counter.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    id_map = {old: idx for idx, (old, _) in enumerate(ordered)}

    lines: List[str] = []
    lines.append("# Company Name Clusters\n")
    lines.append(f"Total clusters: {len(ordered)}\n")

    for old_cid, counter in ordered:
        cid = id_map[old_cid]
        members = sorted(counter.keys(), key=lambda n: (-counter[n], n.lower()))
        size_u = len(members); size_t = sum(counter.values())
        lines.append(f"Cluster {cid} (unique: {size_u}, total: {size_t})")
        for name in members:
            lines.append(f"  - {name}  x{counter[name]}")
        lines.append("")

    # misc cluster (not printed as singleton later)
    if misc_counter and len(misc_counter) > 0:
        members = sorted(misc_counter.keys(), key=lambda n: (-misc_counter[n], n.lower()))
        size_u = len(misc_counter); size_t = sum(misc_counter.values())
        lines.append(f"Cluster misc (unique: {size_u}, total: {size_t})")
        for name in members:
            lines.append(f"  - {name}  x{misc_counter[name]}")
        lines.append("")

    # singletons (excluding misc)
    for lab, name in zip(labels, names):
        if lab < 0 and lab != MISC_LABEL:
            cnt = sum(counts.get(name, Counter()).values())
            lines.append(f"Cluster {name} (singleton)")
            lines.append(f"  - {name}  x{cnt}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_review_pairs(out_path: Path,
                       pairs: List[Tuple[int,int,float,float]],
                       probs: np.ndarray,
                       names: List[str],
                       feats: np.ndarray,
                       k: int) -> None:
    """
    Write top-k borderline pairs (closest to 0.5 probability) for human review.
    Columns: left,right,prob,is_acronym_bridge,semantic_cos,char_cos,token_overlap,specific_jaccard,label
    The 'label' column is pre-created (empty) for you to fill with 1 or 0.
    """
    if k <= 0 or len(pairs) == 0:
        return
    closeness = np.abs(probs - 0.5)
    order = np.argsort(closeness)[:k]
    rows = []
    for idx in order:
        i, j, s_cos, s_chr = pairs[idx]
        rows.append({
            "left": names[i],
            "right": names[j],
            "prob": float(probs[idx]),
            "is_acronym_bridge": int(feats[idx, 10] > 0),
            "semantic_cos": float(s_cos),
            "char_cos": float(s_chr),
            "token_overlap": int(feats[idx, 3]),
            "specific_jaccard": float(feats[idx, 2]),
            "label": ""  # <-- ready for you to fill: 1 (same) or 0 (different)
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

# ----------------------------- Pipeline per file ----------------------------- #

def process_file(csv_path: Path,
                 out_dir: Path,
                 review_dir: Path,
                 columns: List[str],
                 verifier: 'PersistentVerifier',
                 feedback_csv: Optional[Path],
                 emit_review: int) -> Optional[Path]:
    col = pick_column_for_file(csv_path, columns)
    if not col:
        print(f"[SKIP] {csv_path.name}: none of the columns present {columns}")
        return None

    print(f"[FILE] {csv_path.name} (column: {col})")
    names, counts = load_counts(csv_path, col)
    if not names:
        print("  ! No names found")
        return None

    # 1) Prepare signals
    dyn_stop = dynamic_stopwords(names, top_frac=0.30)
    generic_bucket = build_generic_bucket(names, dyn_stop, frac=0.15)

    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embs = embed_all(names, enc)
    Xc = char_tfidf(names)

    # 2) Candidates + features
    pairs, toks_per, acr_pairs = build_candidate_pairs(names, embs, Xc, dyn_stop, generic_bucket)
    if not pairs:
        labels = [-1] * len(names)
        misc_counter = Counter()
        out_path = out_dir / (csv_path.stem + ".txt")
        write_txt(out_path, labels, names, counts, misc_counter)
        print(f"  ✓ Wrote {out_path}")
        return out_path

    feats = pair_features(names, embs, Xc, pairs, dyn_stop, generic_bucket, toks_per, acr_pairs)

    # 3) Seeds for self-training
    y_seeds, seed_idxs = build_seed_labels(names, pairs, feats, dyn_stop, min_seed=60)
    X_seeds = feats[seed_idxs]

    # 4) Optional feedback to accelerate learning
    if feedback_csv is not None and feedback_csv.exists():
        X_fb, y_fb = load_feedback(feedback_csv, names, embs, Xc, dyn_stop, generic_bucket, toks_per, acr_pairs)
        if X_fb.shape[0] > 0:
            X_seeds = np.vstack([X_seeds, X_fb])
            y_seeds = np.concatenate([y_seeds, y_fb])

    # 5) Train/update verifier (persistent) and score all pairs
    verifier.bootstrap_or_update(X_seeds=X_seeds, y_seeds=y_seeds)
    probs = verifier.predict_proba(feats)
    t = verifier.threshold()

    # 6) Structural gate: require specific-token overlap OR explicit acronym bridge
    def specific_overlap_ok(i, j) -> bool:
        si = specific_tokens_from_cache(toks_per, generic_bucket, i)
        sj = specific_tokens_from_cache(toks_per, generic_bucket, j)
        return len(si & sj) >= 1

    edges = []
    for idx, (i, j, _, _) in enumerate(pairs):
        if probs[idx] >= t and (specific_overlap_ok(i, j) or ((i, j) in acr_pairs) or ((j, i) in acr_pairs)):
            edges.append((i, j))

    # Light relaxation if graph is empty (rare)
    if not edges and len(pairs) > 0:
        t2 = max(0.47, t - 0.03)
        for idx, (i, j, _, _) in enumerate(pairs):
            if probs[idx] >= t2 and (specific_overlap_ok(i, j) or ((i, j) in acr_pairs) or ((j, i) in acr_pairs)):
                edges.append((i, j))

    # 7) Connected components
    rows, cols = [], []
    for i, j in edges:
        rows.extend([i, j]); cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.uint8)
    G = csr_matrix((data, (rows, cols)), shape=(len(names), len(names)))
    if G.nnz == 0:
        labels = list(range(len(names)))  # all singletons
    else:
        _, comps = connected_components(csgraph=G, directed=False)
        labels = comps.tolist()

    # 8) Purify clusters
    labels = purify_clusters(names, labels, embs, dyn_stop, generic_bucket)

    # 9) Acronym post-pass (strict) + misc
    labels, misc_counter = post_assign_acronyms_strict(names, labels, counts, embs, dyn_stop)

    # 10) Write results
    out_path = out_dir / (csv_path.stem + ".txt")
    write_txt(out_path, labels, names, counts, misc_counter)
    print(f"  ✓ Wrote {out_path}")

    # 11) Optional: emit a review file (to a separate folder if provided)
    if emit_review > 0:
        review_path = review_dir / f"{csv_path.stem}_review_pairs.csv"
        write_review_pairs(review_path, pairs, probs, names, feats, emit_review)
        print(f"  • Review candidates -> {review_path}")

    return out_path

# ----------------------------------- Main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Entity-equivalence clustering with a persistent, self-updating verifier (TXT per CSV)")
    ap.add_argument("--input-dir", required=True, help="Directory with CSV files")
    ap.add_argument("--output-dir", required=True, help="Directory to write TXT files")
    ap.add_argument("--review-dir", default=None, help="Directory to write review CSVs (defaults to --output-dir)")
    ap.add_argument("--columns", required=True, help="Comma-separated list of column names to scan per CSV")
    ap.add_argument("--feedback", default=None, help="CSV of human-labeled pairs (left,right,label)")
    ap.add_argument("--model-dir", default="./model_store/entity_verifier", help="Directory for persistent verifier")
    ap.add_argument("--emit-review", type=int, default=0, help="How many borderline pairs to export for labeling (0=off)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    review_dir = Path(args.review_dir) if args.review_dir else out_dir
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    feedback_csv = Path(args.feedback) if args.feedback else None

    if not in_dir.is_dir():
        raise SystemExit(f"--input-dir not found or not a directory: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    # Persistent verifier (feature dimension fixed at 11)
    verifier = PersistentVerifier(model_dir=args.model_dir, n_features=11)

    csvs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not csvs:
        raise SystemExit(f"No CSV files in {in_dir}")

    print(f"Found {len(csvs)} CSV file(s)")
    done = 0
    for p in csvs:
        if process_file(p, out_dir, review_dir, columns, verifier, feedback_csv, args.emit_review):
            done += 1
    print(f"Done. Processed {done}/{len(csvs)} file(s)")

if __name__ == "__main__":
    main()

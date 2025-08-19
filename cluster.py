#!/usr/bin/env python3
"""
Dynamic Entity-Equivalence Clustering (Cluster-Level Feedback + No-Chaining)
---------------------------------------------------------------------------
- Processes all CSVs in --input-dir; picks the first present column from --columns.
- Proposes compact initial clusters using a NO-CHAINING merge policy (prevents giant catch-all clusters).
- Persists a pairwise verifier that keeps learning from your approvals/rejections across runs.
- Feedback is at the CLUSTER level: a CSV with two columns only: cluster,label
    * cluster: pipe-separated names, e.g. "MIT | Massachusetts Institute of Technology"
              (or a JSON list: ["MIT","Massachusetts Institute of Technology"])
    * label: 1 (approve) or 0 (reject)
- Approved clusters => must-link constraints + positive training.
- Rejected clusters => mines worst internal pairs => cannot-link constraints + negative training.
- Writes one TXT per CSV with clusters and frequencies.
- Optionally exports review CSVs (cluster,label) to a separate folder.

Install
-------
pip install pandas numpy scipy scikit-learn sentence-transformers

Example
-------
python entity_cluster_clusterfb_nochain.py \
  --input-dir ./data \
  --output-dir ./clusters_out \
  --columns name,company,shipper \
  --emit-review 100 \
  --review-dir ./to_label
"""
from __future__ import annotations

import argparse
import json
import math
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
from scipy.sparse.csgraph import connected_components  # (unused in no-chaining but kept for fallback)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ----------------------------- Regex & constants ------------------------------ #

PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
MULTISPACE_RE = re.compile(r"\s+")
ONLY_ALNUM_RE = re.compile(r"^[a-z0-9]+$")

MISC_LABEL = -999_999  # acronym ambiguity bucket

# ----------------------------- Text helpers ---------------------------------- #

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def _preclean(s: str) -> str:
    # normalize common symbol aliases early
    return s.replace("&", " and ").replace("+", " and ")

def tokenize(name: str) -> List[str]:
    s = strip_accents(str(name).lower().strip())
    s = _preclean(s)
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return [t for t in s.split() if t]

def norm_for_embedding(name: str) -> str:
    toks = tokenize(name)
    return " ".join(toks) if toks else strip_accents(str(name).lower())

def tokens_for_acronym(name: str) -> List[str]:
    return tokenize(name)

def acronym_from_tokens(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens if t)

def is_acronym_only(name: str) -> bool:
    s = strip_accents(_preclean(str(name).lower()))
    s = PUNCT_RE.sub("", s)
    return 1 <= len(s) <= 12 and ONLY_ALNUM_RE.match(s) is not None

def acronym_key_for_acronym_only(name: str) -> str:
    raw = strip_accents(_preclean(name.lower()))
    key = re.sub(r"[^a-z0-9]", "", raw)
    key = key.replace("and", "")
    return key.upper()

# ----------------------------- I/O & counts ---------------------------------- #

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

# ----------------------------- Vectors & candidates --------------------------- #

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

def build_candidate_pairs(names: List[str],
                          embs: np.ndarray,
                          Xc: csr_matrix) -> Tuple[
                              List[Tuple[int,int,float,float]],
                              Set[Tuple[int,int]]
                          ]:
    """
    Returns:
      - pairs: (i,j, semantic_cos, char_cos)
      - acr_pairs: explicit acronym bridges (for info)
    """
    n = len(names)
    k = min(max(10, int(math.sqrt(n)) + 3), n)

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

    # Acronym-bridging candidates (extra recall)
    acr_pairs: Set[Tuple[int,int]] = set()
    acr_only_idx = [i for i, n in enumerate(names) if is_acronym_only(n)]
    exp_idx = [i for i in range(n) if i not in acr_only_idx]
    exp_acr: Dict[str, List[int]] = defaultdict(list)
    for i in exp_idx:
        acr = acronym_from_tokens(tokens_for_acronym(names[i])).replace("and","")
        if 1 <= len(acr) <= 12:
            exp_acr[acr.upper()].append(i)
    for i in acr_only_idx:
        key = acronym_key_for_acronym_only(names[i])
        if key in exp_acr:
            for j in exp_acr[key]:
                a, b = (i, j) if i < j else (j, i)
                acr_pairs.add((a, b))
                add_pair(a, b, s_cos=None, s_chr=0.0)

    pairs = sorted([(i, j, cand[(i,j)][0] or 0.0, cand[(i,j)][1] or 0.0) for (i,j) in cand])
    return pairs, acr_pairs

# ----------------------------- Pairwise features ----------------------------- #

def jaccard_all(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    inter = len(sa & sb); union = len(sa | sb)
    return inter / union if union else 0.0

def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def pair_features(names: List[str],
                  embs: np.ndarray,
                  Xc: csr_matrix,
                  pairs: List[Tuple[int,int,float,float]]) -> np.ndarray:
    """
    0 semantic cosine
    1 char cosine
    2 token jaccard (all tokens)
    3 token overlap count (all tokens)
    4 exact same acronym (non-empty)
    5 one acronym equals other's initials
    6 length ratio
    7 ordered-token sequence ratio
    8 raw string sequence ratio
    9 heuristic: (overlap>=1 and (cos>=.60 or char>=.60))
    10 explicit acronym bridge (set later)
    """
    nfeat = 11
    feats = np.zeros((len(pairs), nfeat), dtype=float)

    toks = [tokenize(s) for s in names]
    acrs = [acronym_from_tokens(tokens_for_acronym(s)).upper().replace("AND","") for s in names]
    normed = [norm_for_embedding(s) for s in names]
    lens = [len(strip_accents(_preclean(s.lower()))) for s in names]

    for k, (i, j, s_cos, s_chr) in enumerate(pairs):
        ji = jaccard_all(toks[i], toks[j])
        ov = len(set(toks[i]) & set(toks[j]))

        ai, aj = acrs[i], acrs[j]
        acr_equal_nonempty = int(bool(ai) and bool(aj) and (ai == aj))
        target_j = re.sub(r"[^A-Z0-9]", "", names[j].upper())
        target_i = re.sub(r"[^A-Z0-9]", "", names[i].upper())
        acr_one_exact_of_other = int((bool(ai) and (ai == target_j)) or (bool(aj) and (aj == target_i)))

        lr = min(lens[i], lens[j]) / max(lens[i], lens[j]) if max(lens[i], lens[j]) else 0.0

        ti_str = " ".join(sorted(list(set(toks[i]))))
        tj_str = " ".join(sorted(list(set(toks[j]))))
        tok_seq = seq_ratio(ti_str, tj_str)
        raw_seq = seq_ratio(normed[i], normed[j])

        feats[k, :] = [
            s_cos, s_chr, ji, ov, acr_equal_nonempty, acr_one_exact_of_other,
            lr, tok_seq, raw_seq,
            1.0 if (ov >= 1 and (s_cos >= 0.60 or s_chr >= 0.60)) else 0.0,
            0.0
        ]
    return feats

def mark_acronym_bridges(feats: np.ndarray, pairs: List[Tuple[int,int,float,float]], acr_pairs: Set[Tuple[int,int]]) -> None:
    idx = {(i, j) if i < j else (j, i): k for k, (i, j, _, _) in enumerate(pairs)}
    for a, b in acr_pairs:
        t = (a, b) if a < b else (b, a)
        if t in idx:
            feats[idx[t], 10] = 1.0

# ----------------------------- Persistent verifier --------------------------- #

@dataclass
class VerifierMeta:
    version: int = 1
    n_updates: int = 0
    n_seen_pairs: int = 0
    threshold_: float = 0.5
    t_low_: float = 0.45
    t_high_: float = 0.55
    n_features_: int = 11

class PersistentVerifier:
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
        if len(np.unique(y_true)) < 2:
            t = 0.5
        else:
            fpr, tpr, thr = roc_curve(y_true, probs)
            j = tpr - fpr
            t = float(max(0.5, thr[np.argmax(j)]))
        self.meta.threshold_ = t
        self.meta.t_low_ = max(0.0, t - 0.05)
        self.meta.t_high_ = min(1.0, t + 0.05)
        return t

    def bootstrap_or_update(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)
        if sample_weight is None:
            cls, counts = np.unique(y, return_counts=True)
            weights = {int(c): (counts.sum() / (len(cls) * cnt)) for c, cnt in zip(cls, counts)}
            sample_weight = np.array([weights[int(t)] for t in y], dtype=float)
        self.clf.partial_fit(Xs, y, classes=self._classes, sample_weight=sample_weight)
        probs = self.clf.predict_proba(Xs)[:, 1]
        self._calibrate_threshold(probs, y)
        self.meta.n_updates += 1
        self.meta.n_seen_pairs += int(X.shape[0])
        self._save_all()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    def threshold(self) -> float:
        return self.meta.threshold_

# ----------------------------- Constraints & feedback ------------------------- #

def load_constraints(store_path: Path) -> Dict[str, List[Tuple[str,str]]]:
    if not store_path.exists():
        return {"must": [], "cannot": []}
    try:
        return json.loads(store_path.read_text())
    except Exception:
        return {"must": [], "cannot": []}

def save_constraints(store_path: Path, data: Dict[str, List[Tuple[str,str]]]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(data, indent=2))

def parse_feedback_clusters(feedback_csv: Path) -> List[Tuple[List[str], int]]:
    """
    Expected columns (case-insensitive): cluster,label
    cluster cell can be:
      - 'name1 | name2 | name3'
      - JSON list: ["name1","name2",...]
    Returns list of (names[], label).
    """
    df = pd.read_csv(feedback_csv)
    cols = {c.lower(): c for c in df.columns}
    if "cluster" not in cols or "label" not in cols:
        return []
    C, L = cols["cluster"], cols["label"]
    out = []
    for _, r in df.iterrows():
        raw = r[C]
        lab = int(r[L])
        if pd.isna(raw):
            continue
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                names = json.loads(raw)
                names = [str(x) for x in names if str(x).strip()]
            except Exception:
                names = [s.strip() for s in str(raw).split("|") if s.strip()]
        else:
            names = [s.strip() for s in str(raw).split("|") if s.strip()]
        if len(names) >= 2:
            out.append((names, lab))
    return out

def pairs_from_indices(idxs: List[int]) -> List[Tuple[int,int]]:
    res = []
    for a in range(len(idxs)):
        for b in range(a+1, len(idxs)):
            i, j = idxs[a], idxs[b]
            if i > j: i, j = j, i
            res.append((i, j))
    return res

def features_for_index_pairs(names: List[str], embs: np.ndarray, Xc: csr_matrix,
                             index_pairs: List[Tuple[int,int]]) -> np.ndarray:
    pairs = [(i, j, 0.0, 0.0) for (i, j) in index_pairs]
    feats = pair_features(names, embs, Xc, pairs)
    return feats

# ----------------------------- No-chaining clusterer ------------------------- #

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1
        return True

def cluster_no_chaining(
    n: int,
    pairs: List[Tuple[int,int,float,float]],
    probs: np.ndarray,
    t: float,
    must_pairs: Set[Tuple[int,int]],
    cannot_pairs: Set[Tuple[int,int]],
) -> List[int]:
    """
    Build compact clusters by forbidding chaining merges.
    - Singletons need >=2 supports (or one very strong) into a cluster.
    - Multi-member merges need cross-density >= 0.60 AND every member has a strong tie.
    """
    prob_of = {(min(i,j), max(i,j)): float(probs[k]) for k, (i, j, _, _) in enumerate(pairs)}
    def p(u,v): return prob_of.get((min(u,v), max(u,v)), 0.0)

    scored = sorted([(float(probs[k]), i, j) for k, (i, j, _, _) in enumerate(pairs)], reverse=True)

    uf = UnionFind(n)
    members: Dict[int, Set[int]] = {i: {i} for i in range(n)}
    must = {(min(i,j), max(i,j)) for (i,j) in must_pairs}
    cannot = {(min(i,j), max(i,j)) for (i,j) in cannot_pairs}

    def root(a): return uf.find(a)

    def blocked(ra: int, rb: int) -> bool:
        for u in members[ra]:
            for v in members[rb]:
                if (min(u,v), max(u,v)) in cannot:
                    return True
        return False

    def merge_roots(ra: int, rb: int):
        if uf.union(ra, rb):
            nr = root(ra)
            if nr == ra:
                members[ra] |= members.pop(rb, set())
            else:
                members[rb] |= members.pop(ra, set())

    # 1) Must-links first
    for (i, j) in sorted(must):
        ra, rb = root(i), root(j)
        if ra != rb:
            merge_roots(ra, rb)

    # 2) Greedy, no-chaining merges
    DENSITY_MIN = 0.60
    VERY_STRONG = min(0.95, t + 0.10)

    for prob, i, j in scored:
        if prob < t: break
        ra, rb = root(i), root(j)
        if ra == rb:
            continue
        if (min(i,j), max(i,j)) not in must and blocked(ra, rb):
            continue

        A, B = members[ra], members[rb]
        # Singleton attach rule
        if len(A) == 1 or len(B) == 1:
            if len(A) == 1: u, others = next(iter(A)), B
            else: u, others = next(iter(B)), A
            supports = sum(1 for v in others if p(u, v) >= t)
            maxp = max((p(u, v) for v in others), default=0.0)
            ok = (supports >= 2) or (maxp >= VERY_STRONG)
            if ok:
                merge_roots(ra, rb)
            continue

        # Multi-member cross support
        total = len(A) * len(B)
        supp = 0
        perA = []
        for u in A:
            max_u = 0.0
            for v in B:
                puv = p(u, v)
                if puv >= t: supp += 1
                if puv > max_u: max_u = puv
            perA.append(max_u)
        perB = []
        for v in B:
            max_v = 0.0
            for u in A:
                puv = p(u, v)
                if puv > max_v: max_v = puv
            perB.append(max_v)

        density = (supp / total) if total else 0.0
        all_have_anchor = (min(perA) >= t) and (min(perB) >= t)
        if density >= DENSITY_MIN and all_have_anchor:
            merge_roots(ra, rb)

    # Final labels by root id
    roots = {i: root(i) for i in range(n)}
    ridx = {}
    nextid = 0
    labels = [-1] * n
    for i, r in roots.items():
        if r not in ridx:
            ridx[r] = nextid
            nextid += 1
        labels[i] = ridx[r]
    return labels

# ----------------------------- Output writers -------------------------------- #

def write_txt(out_path: Path,
              labels: List[int],
              names: List[str],
              counts: Dict[str, Counter],
              misc_counter: Optional[Counter] = None) -> None:
    cluster_to_counter: Dict[int, Counter] = defaultdict(Counter)
    for lab, name in zip(labels, names):
        if lab >= 0:
            cluster_to_counter[lab].update(counts.get(name, Counter()))
    ordered = sorted(cluster_to_counter.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    id_map = {old: idx for idx, (old, _) in enumerate(ordered)}
    lines: List[str] = []
    lines.append("# Clusters\n")
    lines.append(f"Total clusters: {len(ordered)}\n")
    for old_cid, counter in ordered:
        cid = id_map[old_cid]
        members = sorted(counter.keys(), key=lambda n: (-counter[n], n.lower()))
        size_u = len(members); size_t = sum(counter.values())
        lines.append(f"Cluster {cid} (unique: {size_u}, total: {size_t})")
        for name in members:
            lines.append(f"  - {name}  x{counter[name]}")
        lines.append("")
    if misc_counter and len(misc_counter) > 0:
        members = sorted(misc_counter.keys(), key=lambda n: (-misc_counter[n], n.lower()))
        size_u = len(misc_counter); size_t = sum(misc_counter.values())
        lines.append(f"Cluster misc (unique: {size_u}, total: {size_t})")
        for name in members:
            lines.append(f"  - {name}  x{misc_counter[name]}")
        lines.append("")
    for lab, name in zip(labels, names):
        if lab < 0 and lab != MISC_LABEL:
            cnt = sum(counts.get(name, Counter()).values())
            lines.append(f"Cluster {name} (singleton)")
            lines.append(f"  - {name}  x{cnt}")
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def write_review_clusters(out_path: Path,
                          labels: List[int],
                          names: List[str],
                          probs_matrix: Optional[np.ndarray],
                          k: int) -> None:
    """
    Export up to k clusters for human approval with just two columns: cluster,label
    Selection favors low-cohesion or mid-confidence clusters.
    """
    if k <= 0:
        return
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab >= 0:
            clusters[lab].append(i)
    scored: List[Tuple[float, int]] = []
    for cid, idxs in clusters.items():
        if len(idxs) < 2:
            continue
        if probs_matrix is not None:
            sub = probs_matrix[np.ix_(idxs, idxs)]
            tril = sub[np.tril_indices_from(sub, k=-1)]
            if tril.size == 0:
                continue
            min_p = float(np.min(tril))
            avg_p = float(np.mean(tril))
            score = 0.6*(1.0 - min_p) + 0.4*(1.0 - avg_p)  # higher => worse (review first)
        else:
            score = 0.5
        scored.append((score, cid))
    scored.sort(reverse=True)
    rows = []
    for _, cid in scored[:k]:
        idxs = clusters[cid]
        cluster_names = " | ".join(sorted([names[i] for i in idxs], key=lambda s: s.lower()))
        rows.append({"cluster": cluster_names, "label": ""})
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

# ----------------------------- Acronym post-pass ------------------------------ #

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

def post_assign_acronyms_strict(names: List[str],
                                labels: List[int],
                                counts: Dict[str, Counter],
                                embs: np.ndarray) -> Tuple[List[int], Counter]:
    votes: Dict[str, Counter] = defaultdict(Counter)
    acr_to_clusters: Dict[str, Set[int]] = defaultdict(set)
    for idx, (lab, name) in enumerate(zip(labels, names)):
        if lab < 0: continue
        acr = acronym_from_tokens(tokens_for_acronym(name)).upper().replace("AND","")
        if 1 <= len(acr) <= 12:
            votes[acr][lab] += 1
            acr_to_clusters[acr].add(lab)

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
            cen = cents.get(cid, None)
            sim_ok = (cen is not None) and (cosine_sim(embs[i], cen) >= acr_sim_gate)
            if sim_ok:
                labels[i] = cid
                continue
        if len(cset) > 1 or votes.get(key, Counter()).total() > 0:
            labels[i] = MISC_LABEL
            misc.update({names[i]: 1})
        # else remain singleton
    return labels, misc

# ----------------------------- Pipeline per file ----------------------------- #

def process_file(csv_path: Path,
                 out_dir: Path,
                 review_dir: Path,
                 columns: List[str],
                 verifier: 'PersistentVerifier',
                 feedback_csv: Optional[Path],
                 constraints_store: Path,
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

    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embs = embed_all(names, enc)
    Xc = char_tfidf(names)

    pairs, acr_pairs = build_candidate_pairs(names, embs, Xc)
    if not pairs:
        labels = [-1] * len(names)
        misc_counter = Counter()
        out_path = out_dir / (csv_path.stem + ".txt")
        write_txt(out_path, labels, names, counts, misc_counter)
        print(f"  ✓ Wrote {out_path}")
        return out_path

    feats = pair_features(names, embs, Xc, pairs)
    mark_acronym_bridges(feats, pairs, acr_pairs)

    # Self-seeded pairs
    pos_idx, neg_idx = [], []
    for k, (_, _, s_cos, s_chr) in enumerate(pairs):
        ji = feats[k, 2]; ov = feats[k, 3]
        acr_equal = bool(feats[k, 4]); acr_initial = bool(feats[k, 5])
        strong_lex = (ov >= 2 and s_chr >= 0.80) or (ji >= 0.85)
        if acr_equal or acr_initial or strong_lex:
            pos_idx.append(k)
        if ov == 0 and feats[k,0] <= 0.35 and feats[k,1] <= 0.35:
            neg_idx.append(k)
    seed_sel = sorted(set(pos_idx[:60]) | set(neg_idx[:60]))
    X_seeds = feats[seed_sel]
    y_seeds = np.array([1 if s in pos_idx else 0 for s in seed_sel], dtype=int)
    w_seeds = np.ones_like(y_seeds, dtype=float)

    # Constraints store
    constraints_path = constraints_store
    stored = load_constraints(constraints_path)
    must_pairs_str = set(map(tuple, stored.get("must", [])))
    cannot_pairs_str = set(map(tuple, stored.get("cannot", [])))
    idx_by_name = {n: i for i, n in enumerate(names)}
    def map_pairs(pair_strs: Set[Tuple[str,str]]) -> Set[Tuple[int,int]]:
        out = set()
        for a, b in pair_strs:
            if a in idx_by_name and b in idx_by_name:
                i, j = idx_by_name[a], idx_by_name[b]
                out.add((i, j) if i < j else (j, i))
        return out
    must_pairs_idx = map_pairs(must_pairs_str)
    cannot_pairs_idx = map_pairs(cannot_pairs_str)

    # ---- Cluster-level feedback ingestion ----
    fb_pairs_pos: List[Tuple[int,int]] = []
    fb_pairs_neg: List[Tuple[int,int]] = []
    if feedback_csv is not None and feedback_csv.exists():
        fb_clusters = parse_feedback_clusters(feedback_csv)
        # quick warm model to score pairs for negative mining
        ver_warm = PersistentVerifier(model_dir=str(Path(verifier.dir)/"_warm"), n_features=feats.shape[1])
        ver_warm.bootstrap_or_update(X_seeds, y_seeds, sample_weight=w_seeds)
        warm_probs = ver_warm.predict_proba(feats)
        idx_map = {(min(i,j), max(i,j)): k for k, (i, j, _, _) in enumerate(pairs)}

        for names_list, lab in fb_clusters:
            idxs_present = [idx_by_name[n] for n in names_list if n in idx_by_name]
            if len(idxs_present) < 2:
                continue
            ipairs = pairs_from_indices(sorted(set(idxs_present)))
            if lab == 1:
                fb_pairs_pos.extend(ipairs)
                for (i, j) in ipairs:
                    must_pairs_str.add((names[i], names[j]))
            else:
                pair_probs = []
                for (i, j) in ipairs:
                    key = (min(i,j), max(i,j))
                    if key in idx_map:
                        p = warm_probs[idx_map[key]]
                    else:
                        f = features_for_index_pairs(names, embs, Xc, [(i, j)])
                        p = float(ver_warm.predict_proba(f)[0])
                    pair_probs.append(((i, j), p))
                pair_probs.sort(key=lambda x: x[1])
                m = max(1, int(math.ceil(0.2 * len(pair_probs))))
                worst = [ij for (ij, _) in pair_probs[:m]]
                fb_pairs_neg.extend(worst)
                for (i, j) in worst:
                    cannot_pairs_str.add((names[i], names[j]))

        stored["must"] = sorted(list(must_pairs_str))
        stored["cannot"] = sorted(list(cannot_pairs_str))
        save_constraints(constraints_path, stored)

        must_pairs_idx = map_pairs(must_pairs_str)
        cannot_pairs_idx = map_pairs(cannot_pairs_str)

    # Build training set: seeds + feedback-derived
    def feats_for_pairs(pair_list: List[Tuple[int,int]]) -> np.ndarray:
        if not pair_list: return np.empty((0, feats.shape[1]))
        return features_for_index_pairs(names, embs, Xc, pair_list)

    X_fb_pos = feats_for_pairs(fb_pairs_pos)
    y_fb_pos = np.ones((X_fb_pos.shape[0],), dtype=int)
    X_fb_neg = feats_for_pairs(fb_pairs_neg)
    y_fb_neg = np.zeros((X_fb_neg.shape[0],), dtype=int)

    X_train = np.vstack([X_seeds, X_fb_pos, X_fb_neg]) if (X_fb_pos.size or X_fb_neg.size) else X_seeds
    y_train = np.concatenate([y_seeds, y_fb_pos, y_fb_neg]) if (X_fb_pos.size or X_fb_neg.size) else y_seeds
    w_train = np.concatenate([w_seeds,
                              np.full_like(y_fb_pos, 4.0, dtype=float),
                              np.full_like(y_fb_neg, 4.0, dtype=float)]) if (X_fb_pos.size or X_fb_neg.size) else w_seeds

    verifier.bootstrap_or_update(X_train, y_train, sample_weight=w_train)
    probs = verifier.predict_proba(feats)
    t = verifier.threshold()

    # --- No-chaining clustering (prevents giant cluster 0) ---
    labels = cluster_no_chaining(
        n=len(names),
        pairs=pairs,
        probs=probs,
        t=t,
        must_pairs=must_pairs_idx,
        cannot_pairs=cannot_pairs_idx,
    )

    # Purify by centroid similarity (protect must-linked nodes)
    cents = compute_centroids(labels, embs)
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab >= 0:
            cluster_members[lab].append(i)
    sims_all = []
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab); 
        if cen is None: continue
        for i in idxs:
            sims_all.append(cosine_sim(embs[i], cen))
    sim_gate = max(0.45, float(np.percentile(sims_all, 20))) if sims_all else 0.5

    must_nodes = set([u for (u,v) in must_pairs_idx] + [v for (u,v) in must_pairs_idx])
    new_labels = labels[:]
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab)
        for i in idxs:
            if i in must_nodes:
                continue
            sim_ok = (cen is not None) and (cosine_sim(embs[i], cen) >= sim_gate)
            if not sim_ok:
                new_labels[i] = -1
    labels = new_labels

    # Acronym post-pass
    labels, misc_counter = post_assign_acronyms_strict(names, labels, counts, embs)

    # TXT output
    out_path = out_dir / (csv_path.stem + ".txt")
    write_txt(out_path, labels, names, counts, misc_counter)
    print(f"  ✓ Wrote {out_path}")

    # Review CSV (cluster,label only)
    if emit_review > 0:
        N = len(names)
        prob_mat = np.eye(N)
        idx_map = {(min(i,j), max(i,j)): k for k, (i, j, _, _) in enumerate(pairs)}
        for (i, j), k in idx_map.items():
            prob_mat[i, j] = prob_mat[j, i] = probs[k]
        review_path = review_dir / f"{csv_path.stem}_review_clusters.csv"
        write_review_clusters(review_path, labels, names, prob_mat, emit_review)
        print(f"  • Review clusters -> {review_path}")

    return out_path

# ----------------------------------- Main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Dynamic clustering with cluster-level feedback (no-chaining initial merge)")
    ap.add_argument("--input-dir", required=True, help="Directory with CSV files")
    ap.add_argument("--output-dir", required=True, help="Directory to write TXT files")
    ap.add_argument("--columns", required=True, help="Comma-separated list of column names to scan per CSV")
    ap.add_argument("--feedback", default=None, help="CSV of cluster-level labels (columns: cluster,label)")
    ap.add_argument("--emit-review", type=int, default=0, help="How many clusters to export for labeling (0=off)")
    ap.add_argument("--review-dir", default=None, help="Directory to write review CSVs (defaults to --output-dir)")
    ap.add_argument("--model-dir", default="./model_store/entity_verifier", help="Directory for persistent verifier")
    ap.add_argument("--constraints-dir", default=None, help="Path to constraints.json (defaults to <model-dir>/constraints.json)")
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

    if args.constraints_dir:
        constraints_store = Path(args.constraints_dir)
        if constraints_store.is_dir():
            constraints_store = constraints_store / "constraints.json"
    else:
        constraints_store = Path(args.model_dir) / "constraints.json"

    verifier = PersistentVerifier(model_dir=args.model_dir, n_features=11)

    csvs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not csvs:
        raise SystemExit(f"No CSV files in {in_dir}")

    print(f"Found {len(csvs)} CSV file(s)")
    done = 0
    for p in csvs:
        if process_file(p, out_dir, review_dir, columns, verifier, feedback_csv, constraints_store, args.emit_review):
            done += 1
    print(f"Done. Processed {done}/{len(csvs)} file(s)")

if __name__ == "__main__":
    main()

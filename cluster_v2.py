#!/usr/bin/env python3
"""
Entity-Equivalence Clusterer — Dynamic, Domain-Agnostic (TXT per CSV)

Goal
-----
Cluster strings that refer to the SAME real-world entity (entity equivalence),
robustly and dynamically across domains (schools, companies, products, codes).
No manual thresholds; everything adapts to the data.

Signals
-------
1) Semantic embeddings + dynamic HDBSCAN (size-aware).
2) Character 3–5-gram TF-IDF nearest-neighbor graph with an auto-learned
   similarity cutoff (75th percentile of strongest neighbor sims).
3) Graph union → connected components.
4) Purification: eject members that (a) lack informative-token overlap with
   cluster “core” AND (b) are not close enough to the centroid (dynamic sim).
5) Acronym-strict post-pass: attach acronym only if its expansions all land in
   ONE cluster and similarity to that centroid clears a dynamic gate; otherwise
   send to Cluster “misc”.

Output
------
For each CSV: a TXT listing each cluster and member names with frequencies.
Noise/singletons render as individual “Cluster <name> (singleton)”.
Ambiguous acronyms render in “Cluster misc”.

Usage
-----
pip install pandas numpy scipy scikit-learn sentence-transformers hdbscan

python entity_equivalence_clusterer_dynamic_final.py \
  --input-dir ./data \
  --output-dir ./clusters_out \
  --columns Name,Company,Shipper
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Vector / clustering libs
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ------------------------- Text & token utilities ------------------------- #

PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
MULTISPACE_RE = re.compile(r"\s+")
ONLY_ALPHA_RE = re.compile(r"^[a-z]+$")

BASE_CONNECTORS = {
    "of","the","and","at","for","in","&","de","del","la","el","los","las"
}

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

def norm_for_embedding(name: str) -> str:
    # Keep all tokens (after symbol normalization) for semantic context
    toks = tokenize(name)
    return " ".join(toks) if toks else strip_accents(str(name).lower())

def dynamic_stopwords(names: List[str], top_frac: float = 0.30) -> Set[str]:
    """
    Build a data-driven stopword set:
      - tokens that appear in > top_frac of names
      - plus short tokens (<=2 chars)
      - plus base connectors
    """
    df = Counter()
    seen = 0
    for s in names:
        seen += 1
        df.update(set(tokenize(s)))
    hi = {t for t, c in df.items() if c >= top_frac * max(1, seen)}
    short = {t for t in df if len(t) <= 2}
    return hi | short | BASE_CONNECTORS

def informative_tokens(name: str, dyn_stop: Set[str]) -> Set[str]:
    return {t for t in tokenize(name) if t not in dyn_stop and len(t) >= 3}

def tokens_for_acronym(name: str, dyn_stop: Set[str]) -> List[str]:
    toks = tokenize(name)
    return [t for t in toks if t not in dyn_stop]

def acronym_from_tokens(tokens: Sequence[str]) -> str:
    return "".join(t[0] for t in tokens if t)

def is_acronym_only(name: str) -> bool:
    # Strict: letters only; keep 2..8 chars. We normalize symbols before stripping.
    s = strip_accents(_preclean(str(name).lower()))
    s = PUNCT_RE.sub("", s)
    return 2 <= len(s) <= 8 and ONLY_ALPHA_RE.match(s) is not None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

# ---------------------------- Data I/O helpers ---------------------------- #

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

# --------------------------- Dynamic HDBSCAN step ------------------------- #

def choose_hdbscan_params(n: int) -> Tuple[int, int]:
    """
    Size-aware, conservative defaults (no CLI tuning).
    Singletons allowed later as we treat noise as its own cluster.
    """
    if n <= 50:      mcs, ms = 2, 1
    elif n <= 200:   mcs, ms = 3, 1
    elif n <= 1000:  mcs, ms = 5, 3
    else:            mcs, ms = max(6, int(round(np.log2(n)))), max(2, int(round(np.log2(n))) - 1)
    return mcs, ms

def embed_names(names: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode([norm_for_embedding(n) for n in names],
                        normalize_embeddings=True, show_progress_bar=False)

def cluster_with_hdbscan(names: List[str], model: SentenceTransformer) -> Tuple[List[int], np.ndarray]:
    embs = embed_names(names, model)
    mcs, ms = choose_hdbscan_params(len(names))
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean", cluster_selection_method="eom")
    labels = cl.fit_predict(embs)  # -1 = noise
    return labels.tolist(), embs

# --------------------- Dynamic char n-grams graph step -------------------- #

def char_graph(names: List[str]) -> csr_matrix:
    """
    Build a radius graph with an auto cutoff:
      1) Vectorize char 3–5 TF-IDF
      2) Find 10-NN (cosine). Let s_i = best neighbor similarity for i.
      3) Threshold t = 75th percentile of s_i (robust to noise/outliers).
      4) Connect pairs with sim >= t.
    """
    if len(names) <= 1:
        return csr_matrix((len(names), len(names)), dtype=np.uint8)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), lowercase=True, min_df=1, norm="l2")
    X = vec.fit_transform([strip_accents(_preclean(n.lower())) for n in names])

    k = min(10, len(names))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)
    sims = 1.0 - dists[:, 1:].max(axis=1) if k > 1 else np.zeros(len(names))
    t = np.percentile(sims, 75) if np.isfinite(sims).all() else 0.0

    radius = max(0.0, 1.0 - float(t))
    G = radius_neighbors_graph(X, radius=radius, metric="cosine", mode="connectivity", include_self=True)
    return G.tocsr()

# ------------------------------ Graph fusion ------------------------------ #

def labels_from_graph(G: csr_matrix) -> List[int]:
    _, labels = connected_components(csgraph=G, directed=False)
    return labels.tolist()

def graph_union_from_sources(names: List[str], model: SentenceTransformer) -> Tuple[List[int], np.ndarray]:
    # Embedding clusters → adjacency (block-diagonal); Char graph → adjacency; Union → components
    h_labels, embs = cluster_with_hdbscan(names, model)

    # Build adjacency from embedding labels (connect items with the same label >= 0)
    n = len(names)
    if n == 0:
        return [], embs
    rows, cols = [], []
    by_lab: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(h_labels):
        if lab >= 0: by_lab[lab].append(i)
    for _, idxs in by_lab.items():
        for i in idxs:
            for j in idxs:
                rows.append(i); cols.append(j)
    data = np.ones(len(rows), dtype=np.uint8)
    Ge = csr_matrix((data, (rows, cols)), shape=(n, n))

    Gc = char_graph(names)

    G = (Ge + Gc).astype(bool).astype(np.uint8).tocsr()
    comp_labels = labels_from_graph(G)
    return comp_labels, embs

# ---------------------------- Purification step --------------------------- #

def compute_centroids(labels: List[int], embs: np.ndarray) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    agg: Dict[int, List[np.ndarray]] = defaultdict(list)
    for lab, v in zip(labels, embs):
        if lab >= 0:
            agg[lab].append(v)
    for lab, vs in agg.items():
        cents[lab] = np.mean(np.vstack(vs), axis=0)
    return cents

def purify_clusters(
    names: List[str],
    labels: List[int],
    embs: np.ndarray,
    dyn_stop: Set[str],
) -> List[int]:
    """
    Keep a member in its cluster only if:
      (a) it shares at least one informative token with the cluster core, AND
      (b) its embedding similarity to the cluster centroid >= dynamic gate.
    Else → singleton (-1). Gate = 20th percentile of member-to-centroid sims,
    computed across all clusters (floored at 0.38 for safety).
    """
    n = len(names)
    cents = compute_centroids(labels, embs)

    # Per-cluster informative token cores (tokens occurring in >= 2 members)
    core_tokens: Dict[int, Set[str]] = {}
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab >= 0:
            cluster_members[lab].append(i)

    for lab, idxs in cluster_members.items():
        freq = Counter()
        for i in idxs:
            freq.update(informative_tokens(names[i], dyn_stop))
        core_tokens[lab] = {t for t, c in freq.items() if c >= 2}

    # Gather sim distribution to set a dynamic gate
    sims_all = []
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab, None)
        if cen is None: continue
        for i in idxs:
            sims_all.append(cosine_sim(embs[i], cen))
    sim_gate = max(0.38, float(np.percentile(sims_all, 20))) if sims_all else 0.45

    new_labels = labels[:]
    for lab, idxs in cluster_members.items():
        cen = cents.get(lab, None)
        core = core_tokens.get(lab, set())
        for i in idxs:
            has_overlap = len(informative_tokens(names[i], dyn_stop) & core) > 0
            sim_ok = (cen is not None) and (cosine_sim(embs[i], cen) >= sim_gate)
            if not (has_overlap and sim_ok):
                new_labels[i] = -1
    return new_labels

# ----------------------- Acronym-strict post-assignment ------------------- #

def post_assign_acronyms_strict(
    names: List[str],
    labels: List[int],
    counts: Dict[str, Counter],
    embs: np.ndarray,
    dyn_stop: Set[str],
) -> Tuple[List[int], Counter]:
    """
    Attach acronym-only strings to a cluster ONLY when:
      - All observed expansions for that acronym land in ONE cluster
      - That cluster has >= 2 expansion votes for this acronym
      - The acronym embedding is >= dynamic gate from that centroid
    Else → Cluster misc (if there is conflicting evidence) or remain singleton.
    """
    # Expansion votes per acronym per cluster
    votes: Dict[str, Counter] = defaultdict(Counter)
    acr_to_clusters: Dict[str, Set[int]] = defaultdict(set)

    # Build acronym votes from expansions (non-acronym names with labels >= 0)
    for lab, name in zip(labels, names):
        if lab < 0: continue
        acr = acronym_from_tokens(tokens_for_acronym(name, dyn_stop))
        if 2 <= len(acr) <= 8:
            key = acr.upper()
            votes[key][lab] += 1
            acr_to_clusters[key].add(lab)

    # Dynamic sim gate using distribution of member-to-centroid sims
    cents = compute_centroids(labels, embs)
    sims_all = []
    for lab, name in zip(labels, names):
        if lab < 0: continue
        cen = cents.get(lab, None)
        if cen is None: continue
        sims_all.append(cosine_sim(embs[names.index(name)], cen))
    acr_sim_gate = max(0.40, float(np.percentile(sims_all, 25))) if sims_all else 0.45

    # Embed acronyms
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    acr_indices = [i for i, n in enumerate(names) if is_acronym_only(n)]
    acr_vecs = model.encode([norm_for_embedding(names[i]) for i in acr_indices],
                            normalize_embeddings=True, show_progress_bar=False) if acr_indices else np.empty((0,1))

    misc = Counter()
    for j, i in enumerate(acr_indices):
        raw = strip_accents(_preclean(names[i].lower()))
        key = PUNCT_RE.sub("", raw).upper()
        cset = acr_to_clusters.get(key, set())
        if len(cset) == 1:
            cid = next(iter(cset))
            if votes[key][cid] >= 2:  # evidence threshold (dynamic would raise on larger data)
                cen = cents.get(cid, None)
                sim_ok = (cen is not None) and (cosine_sim(acr_vecs[j], cen) >= acr_sim_gate)
                if sim_ok:
                    labels[i] = cid
                    continue
        # Conflicting or weak evidence → misc (not singleton)
        if len(cset) > 1 or votes[key].total() > 0:
            labels[i] = -999999
            misc.update(counts.get(names[i], Counter()))
        # else: no expansions at all → leave as singleton (-1)
    return labels, misc

# ------------------------------- Writer ----------------------------------- #

def write_txt(
    out_path: Path,
    labels: List[int],
    names: List[str],
    counts: Dict[str, Counter],
    misc_counter: Optional[Counter] = None,
) -> None:
    # Positive clusters
    cluster_to_counter: Dict[int, Counter] = defaultdict(Counter)
    for lab, name in zip(labels, names):
        if lab >= 0:
            cluster_to_counter[lab].update(counts.get(name, Counter()))

    # Remap to 0..K-1 by descending size
    ordered = sorted(cluster_to_counter.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    id_map = {old: idx for idx, (old, _) in enumerate(ordered)}

    lines: List[str] = []
    lines.append("# Company Name Clusters\n")
    lines.append(f"Total clusters: {len(ordered)}\n")

    for old_cid, counter in ordered:
        cid = id_map[old_cid]
        members = sorted(counter.keys(), key=lambda n: (-counter[n], n.lower()))
        size_u = len(members); size_t = sum(counter.values())
        lines.append(f"Cluster {cid} (unique: {size_u}, total: {size_t})\n")
        for name in members:
            lines.append(f"  - {name}  x{counter[name]}")
        lines.append("")

    # misc cluster
    if misc_counter and len(misc_counter) > 0:
        members = sorted(misc_counter.keys(), key=lambda n: (-misc_counter[n], n.lower()))
        size_u = len(members); size_t = sum(misc_counter.values())
        lines.append(f"Cluster misc (unique: {size_u}, total: {size_t})\n")
        for name in members:
            lines.append(f"  - {name}  x{misc_counter[name]}")
        lines.append("")

    # singleton/noise clusters
    neg_names = [name for lab, name in zip(labels, names) if lab < 0]
    for name in neg_names:
        cnt = sum(counts.get(name, Counter()).values())
        lines.append(f"Cluster {name} (singleton)\n")
        lines.append(f"  - {name}  x{cnt}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -------------------------------- Runner ---------------------------------- #

def process_file(csv_path: Path, out_dir: Path, columns: List[str]) -> Optional[Path]:
    col = pick_column_for_file(csv_path, columns)
    if not col:
        print(f"[SKIP] {csv_path.name}: none of the columns present {columns}")
        return None

    print(f"[FILE] {csv_path.name} (column: {col})")
    names, counts = load_counts(csv_path, col)
    if not names:
        print("  ! No names found")
        return None

    # Build dynamic stopwords from the actual data
    dyn_stop = dynamic_stopwords(names, top_frac=0.30)

    # Embed + char graph, then union components
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    comp_labels, embs = graph_union_from_sources(names, model)

    # Purify memberships (token-overlap + centroid similarity)
    labels = purify_clusters(names, comp_labels, embs, dyn_stop)

    # Strict acronym assignment (unique+evidence+similarity) with misc quarantine
    labels, misc_counter = post_assign_acronyms_strict(names, labels, counts, embs, dyn_stop)

    out_path = out_dir / (csv_path.stem + ".txt")
    write_txt(out_path, labels, names, counts, misc_counter)
    print(f"  ✓ Wrote {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Dynamic, domain-agnostic entity-equivalence clustering (TXT per CSV)")
    ap.add_argument("--input-dir", required=True, help="Directory with CSV files")
    ap.add_argument("--output-dir", required=True, help="Directory to write TXT files")
    ap.add_argument("--columns", required=True, help="Comma-separated list of column names to scan per CSV")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    if not in_dir.is_dir():
        raise SystemExit(f"--input-dir not found or not a directory: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not csvs:
        raise SystemExit(f"No CSV files in {in_dir}")

    print(f"Found {len(csvs)} CSV file(s)")
    done = 0
    for p in csvs:
        if process_file(p, out_dir, columns):
            done += 1
    print(f"Done. Processed {done}/{len(csvs)} file(s)")

if __name__ == "__main__":
    main()

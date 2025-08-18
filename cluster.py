#!/usr/bin/env python3
"""
Entity‑Equivalence Clustering (TXT per CSV)
-----------------------------------------
From‑scratch, compact implementation focused on **meaning/alias equivalence**
(e.g., "Virginia Tech" ≡ "VT" ≡ "Virginia Polytechnic Institute and State University").

Design goals
- Minimal deps (pandas only). No embeddings, no sklearn.
- Deterministic, transparent rules: token signatures + alias folding + safe
  acronym assignment with a **misc** bucket for ambiguous acronyms.
- Batch over a folder: one TXT per CSV. Required params: --input-dir,
  --output-dir, --columns, --sim-threshold.

Similarity logic (high‑level)
1) Normalize → tokens (lowercased, accents removed, punctuation stripped).
   Drop generic words ("university", "college", "of", "and", ...).
   Alias folding: {polytechnic→tech, (institute & technology)→tech,
   'uc' + location → add 'california', keep location tokens}.
2) Build candidate pairs by rare tokens (blocks). Union when either:
   - share ≥ 2 informative tokens; or
   - share 1 informative token that is long & rare; or
   - trigram Jaccard(name_i, name_j) ≥ --sim-threshold (character safety net).
3) Connected components = clusters.
4) Post‑process acronyms: if an acronym (e.g., "vt", "ucsd") uniquely maps to
   expansions all in a single cluster, attach it there; if it maps to multiple
   clusters, send it to **misc**.

Example
  python entity_equivalence_clusterer.py \
    --input-dir ./data \
    --output-dir ./clusters_out \
    --columns Name,Company,Shipper \
    --sim-threshold 0.46

Install
  pip install pandas
"""
from __future__ import annotations
import argparse
import itertools
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

# ------------------------------ Utilities ------------------------------ #

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
MULTISPACE_RE = re.compile(r"\s+")
ONLY_ALPHA_RE = re.compile(r"^[a-z]+$")

# Generic/stop tokens (keep short; we rely on rare‑token rules for the rest)
STOP = {
    "university", "college", "institutes", "institute", "school",
    "of", "the", "and", "&", "at", "for", "in", "city", "state", "system",
}

# ---------------------------- Normalization ---------------------------- #

def tokenize(name: str) -> List[str]:
    s = strip_accents(str(name).lower().strip())
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    toks = [t for t in s.split() if t]
    return toks


def alias_fold(tokens: List[str]) -> List[str]:
    """Fold common aliases/synonyms without hardcoding specific schools.
    Rules:
      - (institute & technology) → add 'tech' and drop both (to compare with 'X Tech')
      - polytechnic → 'tech'
      - uc + location tokens: keep location, add 'california' (to match 'University of California, <loc>')
    """
    toks = tokens[:]
    ts = set(toks)

    # Handle 'institute of technology' style
    if "institute" in ts and "technology" in ts:
        toks = [t for t in toks if t not in {"institute", "technology"}]
        toks.append("tech")

    # polytechnic → tech
    toks = ["tech" if t == "polytechnic" else t for t in toks]

    # UC <location> → add 'california' (but keep location tokens; don't drop 'uc')
    # If a token is exactly 'uc' and there is any location-ish token present,
    # add 'california' so 'uc san diego' pairs with 'university of california san diego'.
    if "uc" in ts:
        # Heuristic: if there is any token that is not generic/short, treat as location keyword
        locish = [t for t in toks if t not in STOP and len(t) >= 3 and t not in {"uc"}]
        if locish:
            toks.append("california")

    return toks


def informative_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP]


def normalize_tokens(name: str) -> List[str]:
    return informative_tokens(alias_fold(tokenize(name)))

# -------------------------- Trigram similarity -------------------------- #

def trigrams(s: str) -> Set[str]:
    s = strip_accents(s.lower())
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s)
    s = s.replace(" ", "_")  # keep word boundaries
    if len(s) < 3:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}


def jaccard_trigram(a: str, b: str) -> float:
    A, B = trigrams(a), trigrams(b)
    inter = len(A & B)
    if inter == 0:
        return 0.0
    return inter / float(len(A | B))

# --------------------------- Union‑Find (DSU) --------------------------- #

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

# ---------------------------- Core clustering --------------------------- #

@dataclass
class NameRecord:
    original: str
    norm_text: str
    tokens: List[str]


def build_records(unique_names: List[str]) -> List[NameRecord]:
    recs: List[NameRecord] = []
    for name in unique_names:
        toks = normalize_tokens(name)
        norm_text = " ".join(toks) if toks else strip_accents(name.lower())
        recs.append(NameRecord(original=name, norm_text=norm_text, tokens=toks))
    return recs


def acronym_of_tokens(tokens: List[str]) -> str:
    # Ignore very short tokens; keep first letters of meaningful tokens
    parts = [t[0] for t in tokens if len(t) > 1]
    return "".join(parts)


def is_acronym_only(name: str) -> bool:
    s = strip_accents(name.lower()).replace(" ", "")
    return 2 <= len(s) <= 8 and ONLY_ALPHA_RE.match(s) is not None


def cluster_records(recs: List[NameRecord], sim_threshold: float) -> List[int]:
    n = len(recs)
    dsu = DSU(n)

    # Token DF and buckets
    df: Counter = Counter()
    for r in recs:
        df.update(set(r.tokens))

    # Candidate buckets by token (skip overly common tokens)
    names_by_token: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        for t in set(r.tokens):
            if df[t] <= max(3, int(0.15 * n)) and len(t) >= 3:  # rare/medium tokens only
                names_by_token[t].append(i)

    # Pairwise checks within each bucket
    seen_pairs: Set[Tuple[int,int]] = set()
    for t, idxs in names_by_token.items():
        if len(idxs) <= 1:
            continue
        for i, j in itertools.combinations(sorted(idxs), 2):
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            ti, tj = set(recs[i].tokens), set(recs[j].tokens)
            common = ti & tj
            # Rule A: ≥2 shared informative tokens → union
            if len(common) >= 2:
                dsu.union(i, j)
                continue
            # Rule B: 1 shared token that is long & rare → union
            if len(common) == 1:
                tok = next(iter(common))
                if len(tok) >= 6 and df[tok] <= 3:
                    dsu.union(i, j)
                    continue
            # Rule C: Trigram Jaccard fallback
            sim = jaccard_trigram(recs[i].norm_text, recs[j].norm_text)
            if sim >= sim_threshold:
                dsu.union(i, j)

    # Build label array
    roots = [dsu.find(i) for i in range(n)]
    root_to_label: Dict[int, int] = {}
    labels: List[int] = []
    for r in roots:
        if r not in root_to_label:
            root_to_label[r] = len(root_to_label)
        labels.append(root_to_label[r])
    return labels

# ------------------------------ I/O helpers ----------------------------- #

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
            counts[name][name] += 1  # count as seen verbatim
    unique_names = list(counts.keys())
    return unique_names, counts


def write_txt(
    out_path: Path,
    labels: List[int],
    recs: List[NameRecord],
    original_counts: Dict[str, Counter],
    misc_counter: Optional[Counter] = None,
) -> None:
    # Aggregate per cluster
    cluster_to_counter: Dict[int, Counter] = defaultdict(Counter)
    for lab, rec in zip(labels, recs):
        cluster_to_counter[lab].update(original_counts.get(rec.original, Counter()))

    items = sorted(cluster_to_counter.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    lines: List[str] = []
    lines.append("# Company Name Clusters\n")
    lines.append(f"Total clusters: {len(items)}\n")

    for cid, counter in items:
        members = sorted(counter.keys(), key=lambda n: (-counter[n], n.lower()))
        size_u = len(members)
        size_t = sum(counter.values())
        lines.append(f"Cluster {cid} (unique: {size_u}, total: {size_t})\n")
        for name in members:
            lines.append(f"  - {name}  x{counter[name]}")
        lines.append("")

    if misc_counter and len(misc_counter) > 0:
        members = sorted(misc_counter.keys(), key=lambda n: (-misc_counter[n], n.lower()))
        size_u = len(members)
        size_t = sum(misc_counter.values())
        lines.append(f"Cluster misc (unique: {size_u}, total: {size_t})\n")
        for name in members:
            lines.append(f"  - {name}  x{misc_counter[name]}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --------------------------- Acronym assignment ------------------------- #

def assign_acronyms_or_misc(
    recs: List[NameRecord], labels: List[int]
) -> Tuple[List[int], Counter]:
    # Map acronym -> set(cluster_ids) from expansions (non‑acronym names)
    cluster_by_acronym: Dict[str, Set[int]] = defaultdict(set)
    for lab, rec in zip(labels, recs):
        if not is_acronym_only(rec.norm_text):
            acr = acronym_of_tokens(rec.tokens)
            if 2 <= len(acr) <= 8:
                cluster_by_acronym[acr].add(lab)

    # Now assign acronym‑only names
    misc = Counter()
    for i, rec in enumerate(recs):
        if is_acronym_only(rec.norm_text):
            cands = cluster_by_acronym.get(rec.norm_text, set())
            if len(cands) == 1:
                labels[i] = next(iter(cands))
            elif len(cands) > 1:
                # ambiguous → misc bucket (remove from any cluster by marking -1)
                labels[i] = -1
                misc.update({rec.original: 1})
            else:
                # no expansions seen: leave as is (singleton)
                pass
    return labels, misc

# --------------------------------- Runner ------------------------------- #

def process_file(csv_path: Path, out_dir: Path, columns: List[str], sim_threshold: float) -> Optional[Path]:
    col = pick_column_for_file(csv_path, columns)
    if not col:
        print(f"[SKIP] {csv_path.name}: none of the columns present {columns}")
        return None

    print(f"[FILE] {csv_path.name} (column: {col})")
    unique_names, original_counts = load_counts(csv_path, col)
    recs = build_records(unique_names)

    labels = cluster_records(recs, sim_threshold)
    labels, misc_counter = assign_acronyms_or_misc(recs, labels)

    out_path = out_dir / (csv_path.stem + ".txt")
    write_txt(out_path, labels, recs, original_counts, misc_counter)
    print(f"  ✓ Wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Entity‑equivalence clustering (TXT per CSV)")
    ap.add_argument("--input-dir", required=True, help="Directory with CSV files")
    ap.add_argument("--output-dir", required=True, help="Directory to write TXT files")
    ap.add_argument("--columns", required=True, help="Comma‑separated list of column names to scan per CSV")
    ap.add_argument("--sim-threshold", required=True, type=float, help="Trigram Jaccard threshold in [0,1] (fallback matcher)")
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
        if process_file(p, out_dir, columns, args.sim_threshold):
            done += 1
    print(f"Done. Processed {done}/{len(csvs)} file(s)")


if __name__ == "__main__":
    main()

import Papa from "papaparse";
import React, { useEffect, useMemo, useState } from "react";

/**
 * =============================================
 * Entity Clustering Workbench (Tailwind-on-CDN)
 * - Same features as before
 * - Auto-loads Tailwind via CDN so utility classes render like the preview
 * =============================================
 * Fix in this revision:
 * - Resolved "Unterminated JSX contents" by adding the missing closing </div>
 *   so all wrapper containers are properly balanced.
 */

// -------------------- Tailwind Loader --------------------
function useTailwindCDN() {
  useEffect(() => {
    if (typeof window === "undefined") return;
    // If already loaded, skip
    if (document.getElementById("tw-cdn")) return;
    const s = document.createElement("script");
    s.id = "tw-cdn";
    s.src = "https://cdn.tailwindcss.com";
    s.defer = true;
    document.head.appendChild(s);
  }, []);
}

// -------------------- Utilities --------------------
class DSU {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.size = Array(n).fill(1);
  }
  find(x) {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }
  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return;
    if (this.size[ra] < this.size[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    this.size[ra] += this.size[rb];
  }
}

const normalizeName = (s) => {
  if (!s) return "";
  const base = typeof s.normalize === "function" ? s.normalize("NFKD") : s;
  const ascii = base.replace(/[\u0300-\u036f]/g, "");
  return ascii
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
};

// Columns we try to auto-detect for IDs / names
const ID_CANDIDATES = [
  /consignee.*local.*duns/i,
  /consignee.*duns/i,
  /consignee.*panjiva.*id/i,
  /shipper.*panjiva.*id/i,
  /panjiva.*id/i,
  /duns/i,
  /tax.?id|tin|ein/i,
  /company.*id/i,
];

const NAME_CANDIDATES = [/consignee.*name/i, /shipper.*name/i, /name/i];

function pickColumns(headers) {
  const ids = [];
  const names = [];
  headers.forEach((h) => {
    if (ID_CANDIDATES.some((r) => r.test(h))) ids.push(h);
    if (NAME_CANDIDATES.some((r) => r.test(h))) names.push(h);
  });
  return { ids: Array.from(new Set(ids)), names: Array.from(new Set(names)) };
}

function frequencyCanonical(values) {
  const counts = new Map();
  for (const v of values) {
    if (!v) continue;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  if (!counts.size) return "";
  const best = Array.from(counts.entries()).sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0].localeCompare(b[0]);
  })[0][0];
  return best;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// A pure function that performs clustering (used by UI and tests)
function clusterRows(rawRows, idCols, nameCols, strictIdsOnly) {
  const n = rawRows.length;
  const dsu = new DSU(n);

  // Build maps for chosen ID columns
  const idMaps = {};
  idCols.forEach((col) => (idMaps[col] = new Map()));

  for (let i = 0; i < n; i++) {
    for (const col of idCols) {
      const raw = String(rawRows[i]?.[col] ?? "").trim();
      if (!raw) continue;
      const key = raw.toLowerCase();
      const m = idMaps[col];
      if (!m.has(key)) m.set(key, []);
      m.get(key).push(i);
    }
  }

  // Union by each ID bucket
  for (const col of idCols) {
    const m = idMaps[col];
    for (const [, idxs] of m) {
      for (let k = 1; k < idxs.length; k++) dsu.union(idxs[0], idxs[k]);
    }
  }

  // Optional exact-normalized-name merge
  if (!strictIdsOnly) {
    const nameKeyToIdxs = new Map();
    for (let i = 0; i < n; i++) {
      const keys = [];
      for (const c of nameCols) keys.push(normalizeName(rawRows[i]?.[c]));
      const key = Array.from(new Set(keys.filter(Boolean))).join("|");
      if (!key) continue;
      if (!nameKeyToIdxs.has(key)) nameKeyToIdxs.set(key, []);
      nameKeyToIdxs.get(key).push(i);
    }
    for (const [, idxs] of nameKeyToIdxs) {
      for (let k = 1; k < idxs.length; k++) dsu.union(idxs[0], idxs[k]);
    }
  }

  // Materialize clusters
  const comp = {};
  for (let i = 0; i < n; i++) {
    const r = dsu.find(i);
    const k = String(r);
    if (!comp[k]) comp[k] = [];
    comp[k].push(i);
  }

  // Canonical names per cluster
  const canonical = {};
  const chooseFrom = nameCols.length ? nameCols : [];
  for (const [root, idxs] of Object.entries(comp)) {
    const pool = [];
    for (const idx of idxs) {
      for (const c of chooseFrom) {
        const norm = normalizeName(rawRows[idx]?.[c]);
        if (norm) pool.push(norm);
      }
    }
    canonical[root] = frequencyCanonical(pool);
  }

  return { clusters: comp, canonicalByCluster: canonical };
}

// Helper to ensure we never surface empty clusters
function filterNonEmptyClusters(clusters) {
  const out = {};
  for (const [cid, idxs] of Object.entries(clusters || {})) {
    if (Array.isArray(idxs) && idxs.length > 0) out[cid] = idxs;
  }
  return out;
}

// Helper to enforce a minimum cluster size (e.g., 2 to hide singletons)
function filterMinSize(clusters, minSize = 2) {
  const out = {};
  for (const [cid, idxs] of Object.entries(clusters || {})) {
    if (Array.isArray(idxs) && idxs.length >= minSize) out[cid] = idxs;
  }
  return out;
}

// -------------------- UI --------------------
export default function App() {
  // Ensure Tailwind utilities are available (loads once)
  useTailwindCDN();

  const [rawRows, setRawRows] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [idCols, setIdCols] = useState([]);
  const [nameCols, setNameCols] = useState([]);
  const [clusters, setClusters] = useState({}); // display clusters (respect min size)
  const [allClusters, setAllClusters] = useState({}); // full clusters for export
  const [clusterCanonical, setClusterCanonical] = useState({});
  const [approval, setApproval] = useState({}); // cid -> 'approved' | 'rejected' | 'pending'
  const [feedback, setFeedback] = useState({}); // cid -> string
  const [busy, setBusy] = useState(false);
  const [strictIdsOnly, setStrictIdsOnly] = useState(true);

  // Dev-test state
  const [testResults, setTestResults] = useState([]);

  const fileChosen = useMemo(() => rawRows.length > 0, [rawRows]);
  const totalClusters = useMemo(
    () => Object.values(clusters).filter((v) => Array.isArray(v) && v.length > 0).length,
    [clusters]
  );
  const totalRows = useMemo(() => rawRows.length, [rawRows]);

  const handleUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (res) => {
        const data = (res.data || []).filter((r) =>
          Object.values(r).some((v) => String(v ?? "").trim() !== "")
        );
        setRawRows(data);
        const hs = res.meta?.fields ?? Object.keys(data[0] ?? {});
        setHeaders(hs);
        const { ids, names } = pickColumns(hs);
        setIdCols(ids);
        setNameCols(names);
        setClusters({});
        setClusterCanonical({});
        setApproval({});
        setFeedback({});
      },
      error: (err) => {
        console.error(err);
        alert("Failed to parse CSV: " + err.message);
      },
    });
  };

  const runClustering = () => {
    if (!rawRows.length) return;
    setBusy(true);
    try {
      const { clusters: comp, canonicalByCluster } = clusterRows(
        rawRows,
        idCols,
        nameCols,
        strictIdsOnly
      );

      const nonEmptyMin2 = filterMinSize(comp, 2);
      const appr = {};
      for (const cid of Object.keys(nonEmptyMin2)) appr[cid] = "pending";

      setAllClusters(comp);
      setClusters(nonEmptyMin2);
      setClusterCanonical(canonicalByCluster);
      setApproval(appr);
    } finally {
      setBusy(false);
    }
  };

  const exportCSV = () => {
    if (!Object.keys(clusters).length) return alert("Nothing to export yet.");
    const clusterIdByRow = {};
    for (const [cid, idxs] of Object.entries(allClusters)) {
      if (!Array.isArray(idxs) || idxs.length === 0) continue;
      for (const i of idxs) clusterIdByRow[i] = cid;
    }
    const rowsOut = rawRows.map((r, i) => {
      const cid = clusterIdByRow[i];
      const size = Array.isArray(allClusters[cid]) ? allClusters[cid].length : 0;
      return {
        ...r,
        __cluster_id: cid,
        __cluster_size: size,
        __canonical_name: clusterCanonical[cid] ?? "",
        __approval: approval[cid] ?? "pending",
        __feedback: feedback[cid] ?? "",
      };
    });
    const csv = Papa.unparse(rowsOut, { quotes: false });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    downloadBlob(blob, "clustered_output.csv");
  };

  const toggleCol = (inList, setList, col) => {
    setList((cols) => (cols.includes(col) ? cols.filter((c) => c !== col) : [...cols, col]));
  };

  const ClusterCard = ({ cid, idxs }) => {
    const status = approval[cid] ?? "pending";
    return (
      <div className="rounded-2xl border shadow-sm p-3 bg-white">
        <div className="flex items-center justify-between mb-2">
          <div className="font-semibold text-lg">
            Cluster {cid} <span className="text-xs ml-2 px-2 py-0.5 rounded-full bg-gray-100">{idxs.length} rows</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="px-2 py-0.5 rounded-full border">Canonical: {clusterCanonical[cid] || "(empty)"}</span>
            <span
              className={`px-2 py-0.5 rounded-full border ${
                status === "approved"
                  ? "bg-green-600 text-white border-green-600"
                  : status === "rejected"
                  ? "bg-red-600 text-white border-red-600"
                  : "bg-gray-50"
              }`}
            >
              {status}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-2">
          <button
            className={`text-sm px-3 py-1 rounded border ${
              status === "approved" ? "bg-green-600 text-white border-green-600" : "bg-gray-100"
            }`}
            onClick={() => setApproval((a) => ({ ...a, [cid]: "approved" }))}
          >
            Approve
          </button>
          <button
            className={`text-sm px-3 py-1 rounded border ${
              status === "rejected" ? "bg-red-600 text-white border-red-600" : "bg-gray-100"
            }`}
            onClick={() => setApproval((a) => ({ ...a, [cid]: "rejected" }))}
          >
            Reject
          </button>
        </div>

        <div className="mb-3">
          <label className="text-sm font-medium">Feedback (optional)</label>
          <textarea
            className="mt-1 w-full border rounded p-2 text-sm"
            placeholder="Explain why you approved/rejected or note fixes..."
            value={feedback[cid] ?? ""}
            onChange={(e) => setFeedback((f) => ({ ...f, [cid]: e.target.value }))}
          />
        </div>

        <div className="overflow-x-auto border rounded-xl">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                {headers.map((h) => (
                  <th key={h} className="text-left px-2 py-1 whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {idxs.slice(0, 10).map((i) => (
                <tr key={i} className="odd:bg-gray-50">
                  {headers.map((h) => (
                    <td key={h} className="px-2 py-1 whitespace-nowrap">
                      {String(rawRows[i]?.[h] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {idxs.length > 10 && (
            <div className="text-xs text-gray-500 px-2 py-1">Showing first 10 of {idxs.length} rows in this cluster.</div>
          )}
        </div>
      </div>
    );
  };

  // -------------------- Developer Tests --------------------
  function runDevTests() {
    const tests = [];

    // 1) normalizeName
    tests.push({
      name: "normalizeName collapses punctuation & accents",
      run: () => normalizeName("Café & Co., LLC") === "cafe and co llc",
      expect: '"cafe and co llc"',
      got: normalizeName("Café & Co., LLC"),
    });

    tests.push({
      name: "normalizeName handles blanks",
      run: () => normalizeName("") === "",
      expect: '""',
      got: normalizeName(""),
    });

    // 2) frequencyCanonical (tie -> lexicographic)
    const tieValues = ["acme", "beta", "beta", "acme"];
    const tieCanon = frequencyCanonical(tieValues);
    tests.push({
      name: "frequencyCanonical tie breaks lexicographically",
      run: () => tieCanon === "acme",
      expect: '"acme"',
      got: tieCanon,
    });

    // 3) DSU correctness
    const d = new DSU(4);
    d.union(0, 1);
    d.union(2, 3);
    const dsuOk = d.find(0) === d.find(1) && d.find(0) !== d.find(2);
    tests.push({ name: "DSU unions & finds", run: () => dsuOk, expect: "same-set(0,1) and diff from 2", got: dsuOk ? "ok" : "bad" });

    // 4) Clustering by IDs
    const rowsA = [
      { ConsigneeLocalDUNS: "111", ConsigneeName: "ACME, Inc." },
      { ConsigneeLocalDUNS: "111", ConsigneeName: "Acme Incorporated" },
      { ConsigneeLocalDUNS: "222", ConsigneeName: "Beta LLC" },
    ];
    const resA = clusterRows(rowsA, ["ConsigneeLocalDUNS"], ["ConsigneeName"], true);
    const sizesA = Object.values(resA.clusters)
      .map((v) => v.length)
      .sort((a, b) => b - a);
    tests.push({
      name: "Cluster by same DUNS",
      run: () => sizesA[0] === 2 && sizesA[1] === 1,
      expect: "[2,1]",
      got: JSON.stringify(sizesA),
    });

    // 5) Name-based merge when strictIdsOnly=false
    const rowsB = [
      { ConsigneeLocalDUNS: "", ConsigneeName: "Café & Co." },
      { ConsigneeLocalDUNS: "", ConsigneeName: "Cafe and Co" },
      { ConsigneeLocalDUNS: "", ConsigneeName: "Gamma" },
    ];
    const resB = clusterRows(rowsB, ["ConsigneeLocalDUNS"], ["ConsigneeName"], false);
    const sizesB = Object.values(resB.clusters)
      .map((v) => v.length)
      .sort((a, b) => b - a);
    tests.push({
      name: "Exact normalized-name merge (non-strict)",
      run: () => sizesB[0] === 2 && sizesB[1] === 1,
      expect: "[2,1]",
      got: JSON.stringify(sizesB),
    });

    // 6) Non-empty cluster filter
    const fake = { a: [1, 2], b: [], c: [3] };
    const filtered = filterNonEmptyClusters(fake);
    tests.push({
      name: "filterNonEmptyClusters removes empties",
      run: () => Object.keys(filtered).length === 2 && filtered.b === undefined,
      expect: "clusters a & c remain, b removed",
      got: JSON.stringify(filtered),
    });

    // 7) Min-size filter removes singletons
    const withSingletons = { g1: [0,1], g2: [2], g3: [3,4,5] };
    const min2 = filterMinSize(withSingletons, 2);
    tests.push({
      name: "filterMinSize (>=2) hides singletons",
      run: () => Object.keys(min2).length === 2 && !("g2" in min2),
      expect: "g1 and g3 remain, g2 removed",
      got: JSON.stringify(min2),
    });

    setTestResults(
      tests.map((t) => ({ name: t.name, passed: !!t.run(), expect: t.expect, got: t.got }))
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <div className="px-6 md:px-10 py-6">
        <div className="max-w-7xl mx-auto space-y-6 md:space-y-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Entity Clustering Workbench</h1>
            <div className="flex gap-2">
              <button
                className="px-3 py-2 rounded-lg border bg-white disabled:opacity-50"
                onClick={exportCSV}
                disabled={!totalClusters}
              >
                Download CSV
              </button>
            </div>
          </div>

          <div className="rounded-2xl border bg-white">
            <div className="p-4 border-b">
              <div className="text-lg font-semibold">Upload CSV</div>
            </div>
            <div className="p-4 space-y-4">
              <input type="file" accept=".csv" onChange={handleUpload} className="block w-full text-sm" />

              {fileChosen && (
                <div className="grid sm:grid-cols-2 gap-4">
                  <div className="p-3 border rounded-2xl bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">Identifier Columns</h3>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-white border">{idCols.length} selected</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {headers.map((h) => (
                        <button
                          key={h}
                          onClick={() => toggleCol(idCols, setIdCols, h)}
                          className={`text-xs px-2 py-1 rounded-full border ${
                            idCols.includes(h) ? "bg-blue-600 text-white border-blue-600" : "bg-white"
                          }`}
                          title="Toggle as ID column"
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                    <div className="mt-3 flex items-center gap-2">
                      <input
                        id="strict"
                        type="checkbox"
                        checked={strictIdsOnly}
                        onChange={(e) => setStrictIdsOnly(e.target.checked)}
                      />
                      <label htmlFor="strict" className="text-sm">
                        Strict IDs only (disable to also merge by exact normalized name)
                      </label>
                    </div>
                  </div>

                  <div className="p-3 border rounded-2xl bg-gray-50">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">Name Columns</h3>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-white border">{nameCols.length} selected</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {headers.map((h) => (
                        <button
                          key={h}
                          onClick={() => toggleCol(nameCols, setNameCols, h)}
                          className={`text-xs px-2 py-1 rounded-full border ${
                            nameCols.includes(h) ? "bg-blue-600 text-white border-blue-600" : "bg-white"
                          }`}
                          title="Toggle as Name column"
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center gap-3">
                <button
                  className="px-3 py-2 rounded-lg border bg-white disabled:opacity-50"
                  disabled={!fileChosen || busy}
                  onClick={runClustering}
                >
                  {busy ? "Running..." : "Run Clustering"}
                </button>
                {fileChosen && (
                  <div className="text-sm text-gray-600">
                    Rows: <b>{totalRows}</b> · Clusters: <b>{totalClusters || "-"}</b>
                  </div>
                )}
              </div>
            </div>
          </div>

          {!!totalClusters && (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {Object.entries(clusters)
                .filter(([, idxs]) => Array.isArray(idxs) && idxs.length >= 2)
                .sort((a, b) => b[1].length - a[1].length)
                .map(([cid, idxs]) => (
                  <ClusterCard key={cid} cid={cid} idxs={idxs} />
                ))}
            </div>
          )}

          {!totalClusters && !busy && fileChosen && (
            <div className="text-sm text-gray-600">
              No clusters yet — choose ID/Name columns and click <b>Run Clustering</b>.
            </div>
          )}

          {/* Developer Tests */}
          <div className="rounded-2xl border bg-white">
            <div className="p-4 border-b flex items-center justify-between">
              <div className="text-lg font-semibold">Developer Tests</div>
              <button className="px-3 py-2 rounded-lg border bg-white" onClick={runDevTests}>
                Run Dev Tests
              </button>
            </div>
            <div className="p-4">
              {testResults.length === 0 ? (
                <div className="text-sm text-gray-600">
                  Click <b>Run Dev Tests</b> to validate core functions (normalization, canonical selection, DSU, and clustering).
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="text-left px-2 py-1">Test</th>
                      <th className="text-left px-2 py-1">Passed</th>
                      <th className="text-left px-2 py-1">Expected</th>
                      <th className="text-left px-2 py-1">Got</th>
                    </tr>
                  </thead>
                  <tbody>
                    {testResults.map((t, i) => (
                      <tr key={i} className="odd:bg-gray-50">
                        <td className="px-2 py-1">{t.name}</td>
                        <td className={`px-2 py-1 font-medium ${t.passed ? "text-green-700" : "text-red-700"}`}>
                          {t.passed ? "✔" : "✘"}
                        </td>
                        <td className="px-2 py-1">{String(t.expect)}</td>
                        <td className="px-2 py-1">{String(t.got)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

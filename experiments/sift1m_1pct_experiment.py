from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import h5py
import numpy as np
import pandas as pd

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def splitmix64_array(x: np.ndarray) -> np.ndarray:
    z = (x.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)) & MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    return z ^ (z >> np.uint64(31))


def independent_mask(n: int, s: float, seed: int) -> np.ndarray:
    ids = np.arange(n, dtype=np.uint64)
    h = splitmix64_array(ids ^ np.uint64(seed))
    threshold = np.uint64(int(s * (1 << 64)))
    return h < threshold


def l2_batch(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = x - q
    return np.einsum("ij,ij->i", d, d, optimize=True)


def recall10(found: np.ndarray, truth: np.ndarray) -> float:
    if truth.size == 0:
        return 1.0
    return len(set(found[:10].tolist()).intersection(truth[:10].tolist())) / 10.0


def build_hnsw(xb: np.ndarray, m: int, efc: int, threads: int) -> Tuple[object, float]:
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(xb.shape[1], m)
    index.hnsw.efConstruction = efc
    index.hnsw.efSearch = 64
    t0 = time.perf_counter()
    index.add(xb)
    elapsed = time.perf_counter() - t0
    log(f"base HNSW built in {elapsed:.2f}s")
    return index, elapsed


def extract_layer0(index: object, n: int) -> Tuple[np.ndarray, np.ndarray]:
    h = index.hnsw
    levels = faiss.vector_to_array(h.levels).astype(np.int32, copy=False)
    offsets = faiss.vector_to_array(h.offsets).astype(np.int64, copy=False)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int32, copy=False)
    m0 = int(h.nb_neighbors(0))
    adj = np.empty((n, m0), dtype=np.int32)
    cols = np.arange(m0, dtype=np.int64)
    for start in range(0, n, 100_000):
        end = min(n, start + 100_000)
        adj[start:end] = neighbors[offsets[start:end, None] + cols[None, :]]
    log(f"layer0 extracted: width={m0}, actual edges={np.count_nonzero(adj >= 0):,}")
    return levels, adj


def choose_level(levels_raw: np.ndarray, s: float) -> Tuple[int, np.ndarray, pd.DataFrame]:
    max_level = int(levels_raw.max()) - 1
    rows = []
    candidates = []
    n = len(levels_raw)
    for level in range(1, max_level + 1):
        portals = np.flatnonzero(levels_raw >= level + 1).astype(np.int32)
        if len(portals) < 16:
            continue
        pi = len(portals) / n
        score = abs(math.log(pi / s))
        rows.append({"level": level, "n_nodes": len(portals), "sampling_rate": pi, "scale_score": score})
        candidates.append((score, level, portals))
    candidates.sort(key=lambda z: z[0])
    _, level, portals = candidates[0]
    return level, portals, pd.DataFrame(rows)


def assign_geometric_cells(xb: np.ndarray, portals: np.ndarray, threads: int) -> Tuple[np.ndarray, float]:
    faiss.omp_set_num_threads(threads)
    pindex = faiss.IndexHNSWFlat(xb.shape[1], 8)
    pindex.hnsw.efConstruction = 80
    pindex.hnsw.efSearch = 128
    pindex.add(xb[portals])
    cells = np.empty(len(xb), dtype=np.int32)
    t0 = time.perf_counter()
    for start in range(0, len(xb), 25_000):
        end = min(len(xb), start + 25_000)
        _, ids = pindex.search(xb[start:end], 1)
        cells[start:end] = ids[:, 0].astype(np.int32)
    elapsed = time.perf_counter() - t0
    counts = np.bincount(cells, minlength=len(portals))
    log(f"geometric cells assigned in {elapsed:.2f}s; mean={counts.mean():.2f}, median={np.median(counts):.1f}, p99={np.quantile(counts,.99):.1f}")
    return cells, elapsed


def keys_to_csr(keys: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    if keys.size == 0:
        return np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
    keys = np.unique(keys.astype(np.int64, copy=False))
    src = (keys // n_cells).astype(np.int32, copy=False)
    dst = (keys % n_cells).astype(np.int32, copy=False)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    counts = np.bincount(src, minlength=n_cells)
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, dst.copy()


def contract_graph(adj: np.ndarray, cells: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray, float]:
    pieces: List[np.ndarray] = []
    t0 = time.perf_counter()
    for start in range(0, len(adj), 100_000):
        end = min(len(adj), start + 100_000)
        block = adj[start:end]
        valid = block >= 0
        dst_ids = block[valid]
        src_cells = np.broadcast_to(cells[start:end, None], block.shape)[valid]
        dst_cells = cells[dst_ids]
        cross = src_cells != dst_cells
        if np.any(cross):
            keys = src_cells[cross].astype(np.int64) * n_cells + dst_cells[cross].astype(np.int64)
            pieces.append(np.unique(keys))
    keys = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
    src, dst = keys // n_cells, keys % n_cells
    keys = np.unique(np.concatenate([keys, dst * n_cells + src]))
    indptr, indices = keys_to_csr(keys, n_cells)
    elapsed = time.perf_counter() - t0
    log(f"full quotient: edges={len(indices):,}, degree={len(indices)/n_cells:.2f}, {elapsed:.2f}s")
    return indptr, indices, elapsed


def bottomk_graph(indptr: np.ndarray, indices: np.ndarray, n_cells: int, b: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    src_parts, dst_parts = [], []
    for u in range(n_cells):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        if len(nb) <= b:
            chosen = nb
        else:
            x = nb.astype(np.uint64) ^ np.uint64(u * 0x9E3779B1) ^ np.uint64(seed)
            ranks = splitmix64_array(x)
            chosen = nb[np.argpartition(ranks, b - 1)[:b]]
        if len(chosen):
            src_parts.append(np.full(len(chosen), u, dtype=np.int32))
            dst_parts.append(chosen.astype(np.int32, copy=False))
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    keys = src.astype(np.int64) * n_cells + dst.astype(np.int64)
    rev = dst.astype(np.int64) * n_cells + src.astype(np.int64)
    out = keys_to_csr(np.concatenate([keys, rev]), n_cells)
    log(f"bottom-{b}: edges={len(out[1]):,}, degree={len(out[1])/n_cells:.2f}")
    return out


def grouped_ids(cells: np.ndarray, mask: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.flatnonzero(mask).astype(np.int32)
    labels = cells[ids]
    order = np.argsort(labels, kind="stable")
    ids = ids[order]
    counts = np.bincount(labels[order], minlength=n_cells)
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return ids, offsets


def quotient_search(xb: np.ndarray, q: np.ndarray, portals: np.ndarray, indptr: np.ndarray, indices: np.ndarray, entry: int, ef: int, qualified: np.ndarray, qoff: np.ndarray) -> Tuple[np.ndarray, int, int, int, float]:
    t0 = time.perf_counter()
    n_cells = len(portals)
    ef = min(max(1, ef), n_cells)
    visited = np.zeros(n_cells, dtype=np.uint8)
    d0 = float(np.sum((xb[portals[entry]] - q) ** 2))
    cand: List[Tuple[float, int]] = [(d0, entry)]
    top: List[Tuple[float, int]] = [(-d0, entry)]
    visited[entry] = 1
    pcomps, scans = 1, 0
    while cand:
        du, u = heapq.heappop(cand)
        if len(top) >= ef and du > -top[0][0]:
            break
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        scans += len(nb)
        if len(nb) == 0:
            continue
        unseen = nb[visited[nb] == 0]
        if len(unseen) == 0:
            continue
        visited[unseen] = 1
        ds = l2_batch(xb[portals[unseen]], q)
        pcomps += len(unseen)
        for dv, v in zip(ds.tolist(), unseen.tolist()):
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(cand, (float(dv), int(v)))
                heapq.heappush(top, (-float(dv), int(v)))
                if len(top) > ef:
                    heapq.heappop(top)
    selected = [u for _, u in sorted([(-nd, u) for nd, u in top])]
    parts = []
    for c in selected:
        a, z = int(qoff[c]), int(qoff[c + 1])
        if z > a:
            parts.append(qualified[a:z])
    if parts:
        payload = np.concatenate(parts)
        ds = l2_batch(xb[payload], q)
        kk = min(10, len(payload))
        if len(payload) <= kk:
            order = np.argsort(ds)
        else:
            part = np.argpartition(ds, kk - 1)[:kk]
            order = part[np.argsort(ds[part])]
        answer = payload[order]
    else:
        payload = np.empty(0, dtype=np.int32)
        answer = payload
    return answer, pcomps, len(payload), scans, 1000 * (time.perf_counter() - t0)


def exact_truth(xb: np.ndarray, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ds = l2_batch(xb[ids], q)
    kk = min(10, len(ids))
    part = np.argpartition(ds, kk - 1)[:kk]
    return ids[part[np.argsort(ds[part])]]


def estimator_rows(levels_raw: np.ndarray, masks: List[np.ndarray], true_s: float) -> pd.DataFrame:
    rows = []
    max_level = int(levels_raw.max()) - 1
    for seed_idx, mask in enumerate(masks):
        for level in range(0, max_level + 1):
            sample = levels_raw >= level + 1
            n_l = int(sample.sum())
            x_l = int(np.count_nonzero(mask & sample))
            shat = x_l / n_l if n_l else float("nan")
            rows.append({"mask_seed":seed_idx,"level":level,"sample_size":n_l,"matches":x_l,"s_hat":shat,"relative_error":abs(shat-true_s)/true_s if n_l else float("nan"),"nonempty":x_l>0})
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        xb = np.asarray(f["train"], dtype=np.float32)
        xq = np.asarray(f["test"][:args.queries], dtype=np.float32)
    n = len(xb)
    log(f"loaded SIFT1M: {xb.shape}, queries={len(xq)}")
    index, build_s = build_hnsw(xb, args.M, args.ef_construction, args.threads)
    levels_raw, adj = extract_layer0(index, n)
    level, portals, level_df = choose_level(levels_raw, args.selectivity)
    log(f"matched level L{level}: portals={len(portals):,}, rate={len(portals)/n:.6f}")
    cells, assign_s = assign_geometric_cells(xb, portals, args.threads)
    full_i, full_x, contract_s = contract_graph(adj, cells, len(portals))
    graphs = {"full_quotient": (full_i, full_x)}
    for b in (4, 8, 16):
        graphs[f"bottom{b}_quotient"] = bottomk_graph(full_i, full_x, len(portals), b, args.seed + b)
    masks = [independent_mask(n, args.selectivity, args.seed + 1000*i) for i in range(args.mask_seeds)]
    estimator_rows(levels_raw, masks, args.selectivity).to_csv(out / "selectivity_estimator.csv", index=False)
    level_df.to_csv(out / "level_metrics.csv", index=False)
    grouped = [grouped_ids(cells, m, len(portals)) for m in masks]
    truths: Dict[Tuple[int,int], np.ndarray] = {}
    qids = list(range(len(xq)))
    for qid in qids:
        si = qid % args.mask_seeds
        truths[(si,qid)] = exact_truth(xb, xq[qid], grouped[si][0])
    index.hnsw.efSearch = 64
    _, entry_ids = index.search(xq, 1)
    entries = cells[entry_ids[:,0]]
    rows: List[Dict[str,object]] = []
    for qid in qids:
        si = qid % args.mask_seeds
        ids = grouped[si][0]
        t0 = time.perf_counter()
        exact_truth(xb, xq[qid], ids)
        rows.append({"method":"prefilter_exact","query":qid,"mask_seed":si,"parameter":0,"recall":1.0,"distance_computations":len(ids),"portal_computations":0,"payload_computations":len(ids),"edge_scans":0,"latency_ms":1000*(time.perf_counter()-t0),"valid_count":len(ids)})
    for pool in [200,400,1000,2000,4000,5000,10000,20000]:
        index.hnsw.efSearch = pool
        t0 = time.perf_counter()
        _, cand = index.search(xq, pool)
        elapsed = 1000*(time.perf_counter()-t0)/len(xq)
        for qid in qids:
            si = qid % args.mask_seeds
            m = masks[si]
            valid = cand[qid][m[cand[qid]]][:10].astype(np.int32)
            rows.append({"method":"postfilter_hnsw","query":qid,"mask_seed":si,"parameter":pool,"recall":recall10(valid,truths[(si,qid)]),"distance_computations":np.nan,"portal_computations":np.nan,"payload_computations":len(valid),"edge_scans":np.nan,"latency_ms":elapsed,"valid_count":int(m.sum())})
        log(f"postfilter pool={pool} complete")
    efs = [int(x) for x in args.efs.split(',')]
    for method, graph in graphs.items():
        for ef in efs:
            for qid in qids:
                si = qid % args.mask_seeds
                qual, qoff = grouped[si]
                ans, pc, payload, scans, ms = quotient_search(xb,xq[qid],portals,graph[0],graph[1],int(entries[qid]),ef,qual,qoff)
                rows.append({"method":method,"query":qid,"mask_seed":si,"parameter":ef,"recall":recall10(ans,truths[(si,qid)]),"distance_computations":pc+payload,"portal_computations":pc,"payload_computations":payload,"edge_scans":scans,"latency_ms":ms,"valid_count":len(qual)})
            log(f"{method} ef={ef} complete")
    qdf = pd.DataFrame(rows)
    qdf.to_csv(out / "query_results.csv", index=False)
    sdf = qdf.groupby(["method","parameter"],as_index=False).agg(recall=("recall","mean"),recall_p10=("recall",lambda x:float(np.quantile(x,.1))),distance_computations=("distance_computations","mean"),portal_computations=("portal_computations","mean"),payload_computations=("payload_computations","mean"),edge_scans=("edge_scans","mean"),latency_ms=("latency_ms","mean"),valid_count=("valid_count","mean"))
    sdf.to_csv(out / "summary.csv", index=False)
    sm = []
    for si, mask in enumerate(masks):
        valid_ids = np.flatnonzero(mask)
        block = adj[valid_ids]
        deg = np.sum((block >= 0) & mask[np.maximum(block,0)], axis=1)
        active = np.bincount(cells[valid_ids], minlength=len(portals)) > 0
        for method,(ip,ix) in graphs.items():
            act_deg = []
            for u in np.flatnonzero(active):
                nb = ix[int(ip[u]):int(ip[u+1])]
                act_deg.append(int(active[nb].sum()))
            sm.append({"mask_seed":si,"method":method,"point_valid_degree":float(deg.mean()),"point_zero_degree":float(np.mean(deg==0)),"active_cells":int(active.sum()),"active_degree":float(np.mean(act_deg)),"active_zero_degree":float(np.mean(np.asarray(act_deg)==0))})
    pd.DataFrame(sm).to_csv(out / "structural_metrics.csv",index=False)
    meta = {"n":n,"d":xb.shape[1],"queries":len(xq),"selectivity":args.selectivity,"matched_level":level,"portals":len(portals),"build_seconds":build_s,"assignment_seconds":assign_s,"contract_seconds":contract_s,"full_edges":len(full_x),"bottom4_edges":len(graphs["bottom4_quotient"][1]),"bottom8_edges":len(graphs["bottom8_quotient"][1]),"bottom16_edges":len(graphs["bottom16_quotient"][1])}
    (out / "metadata.json").write_text(json.dumps(meta,indent=2))
    log(f"done; outputs={out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",default="data/sift-128-euclidean.hdf5")
    p.add_argument("--out",default="results/sift1m_1pct")
    p.add_argument("--queries",type=int,default=60)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--M",type=int,default=8)
    p.add_argument("--ef-construction",type=int,default=80)
    p.add_argument("--selectivity",type=float,default=0.01)
    p.add_argument("--mask-seeds",type=int,default=3)
    p.add_argument("--seed",type=int,default=20260807)
    p.add_argument("--efs",default="32,64,128,256,512,1024,2048")
    main(p.parse_args())

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import faiss
import h5py
import numpy as np
import pandas as pd

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
UINT64_DENOM = float(2**64)


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
    threshold_int = int(s * (1 << 64))
    if threshold_int >= (1 << 64):
        return np.ones(n, dtype=bool)
    return h < np.uint64(threshold_int)


def l2_batch(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = x - q
    return np.einsum("ij,ij->i", d, d, optimize=True)


def recall_at_10(found: np.ndarray, truth: np.ndarray) -> float:
    if truth.size == 0:
        return 1.0
    return len(set(found[:10].tolist()).intersection(truth[:10].tolist())) / 10.0


def topk_ids(xb: np.ndarray, q: np.ndarray, ids: np.ndarray, k: int = 10) -> np.ndarray:
    if ids.size == 0:
        return np.empty(0, dtype=np.int32)
    ds = l2_batch(xb[ids], q)
    kk = min(k, ids.size)
    if ids.size <= kk:
        order = np.argsort(ds)
    else:
        part = np.argpartition(ds, kk - 1)[:kk]
        order = part[np.argsort(ds[part])]
    return ids[order].astype(np.int32, copy=False)


def build_hnsw(xb: np.ndarray, m: int, efc: int, threads: int) -> Tuple[object, float]:
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(xb.shape[1], m)
    index.hnsw.efConstruction = efc
    index.hnsw.efSearch = 64
    t0 = time.perf_counter()
    index.add(xb)
    elapsed = time.perf_counter() - t0
    log(f"base HNSW built: n={len(xb):,}, M={m}, efC={efc}, seconds={elapsed:.2f}")
    return index, elapsed


def extract_layer0(index: object, n: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
    actual = int(np.count_nonzero(adj >= 0))
    log(f"layer-0 extracted: slots={adj.size:,}, actual edges={actual:,}, width={m0}")
    return levels, adj, actual, int(neighbors.size)


def portal_ids(levels_raw: np.ndarray, level: int) -> np.ndarray:
    return np.flatnonzero(levels_raw >= level + 1).astype(np.int32)


def available_levels(levels_raw: np.ndarray, min_nodes: int = 16) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n = len(levels_raw)
    max_level = int(levels_raw.max()) - 1
    for level in range(1, max_level + 1):
        p = portal_ids(levels_raw, level)
        if len(p) < min_nodes:
            continue
        rows.append(
            {
                "level": level,
                "n_nodes": len(p),
                "sampling_rate": len(p) / n,
                "mean_cell_size_if_balanced": n / len(p),
            }
        )
    return pd.DataFrame(rows)


def choose_target_level(level_df: pd.DataFrame, s: float, allowed: Sequence[int]) -> int:
    candidates = level_df[level_df["level"].isin(list(allowed))].copy()
    candidates["score"] = np.abs(np.log(candidates["sampling_rate"] / s))
    return int(candidates.sort_values("score").iloc[0]["level"])


def build_small_hnsw(x: np.ndarray, threads: int, m: int = 8, efc: int = 80, efs: int = 128) -> object:
    faiss.omp_set_num_threads(threads)
    idx = faiss.IndexHNSWFlat(x.shape[1], m)
    idx.hnsw.efConstruction = efc
    idx.hnsw.efSearch = efs
    idx.add(x)
    return idx


def assign_geometric_cells(
    xb: np.ndarray,
    portals: np.ndarray,
    threads: int,
    batch: int = 25_000,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    pindex = build_small_hnsw(xb[portals], threads=threads, m=8, efc=80, efs=128)
    cells = np.empty(len(xb), dtype=np.int32)
    t0 = time.perf_counter()
    for start in range(0, len(xb), batch):
        end = min(len(xb), start + batch)
        _, ids = pindex.search(xb[start:end], 1)
        cells[start:end] = ids[:, 0].astype(np.int32)
    elapsed = time.perf_counter() - t0
    counts = np.bincount(cells, minlength=len(portals))
    stats = {
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "p90": float(np.quantile(counts, 0.90)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(counts.max()),
    }
    log(
        f"L-cell assignment: portals={len(portals):,}, seconds={elapsed:.2f}, "
        f"mean={stats['mean']:.2f}, median={stats['median']:.1f}, p99={stats['p99']:.1f}"
    )
    return cells, elapsed, stats


def assign_portal_parents(
    xb: np.ndarray,
    child_portals: np.ndarray,
    parent_portals: np.ndarray,
    threads: int,
) -> Tuple[np.ndarray, float]:
    pindex = build_small_hnsw(xb[parent_portals], threads=threads, m=8, efc=80, efs=128)
    t0 = time.perf_counter()
    _, ids = pindex.search(xb[child_portals], 1)
    elapsed = time.perf_counter() - t0
    parent = ids[:, 0].astype(np.int32)
    log(
        f"parent assignment: children={len(child_portals):,}, parents={len(parent_portals):,}, "
        f"seconds={elapsed:.2f}"
    )
    return parent, elapsed


def keys_to_csr(keys: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    if keys.size == 0:
        return np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
    keys = np.unique(keys.astype(np.int64, copy=False))
    src = (keys // n_cells).astype(np.int32, copy=False)
    dst = (keys % n_cells).astype(np.int32, copy=False)
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
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
    src = keys // n_cells
    dst = keys % n_cells
    keys = np.unique(np.concatenate((keys, dst * n_cells + src)))
    indptr, indices = keys_to_csr(keys, n_cells)
    elapsed = time.perf_counter() - t0
    log(
        f"full quotient: cells={n_cells:,}, directed edges={indices.size:,}, "
        f"mean degree={indices.size / max(n_cells, 1):.2f}, seconds={elapsed:.2f}"
    )
    return indptr, indices, elapsed


def edge_sources(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(indptr) - 1, dtype=np.int32), np.diff(indptr))


def edge_randoms(indptr: np.ndarray, indices: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    src = edge_sources(indptr).astype(np.uint64, copy=False)
    dst = indices.astype(np.uint64, copy=False)
    x = dst ^ (src * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)
    rank = splitmix64_array(x)
    u = (rank.astype(np.float64) + 1.0) / (UINT64_DENOM + 1.0)
    neglog = -np.log(u)
    return rank, neglog


def uniform_bottomk_graph(
    indptr: np.ndarray,
    indices: np.ndarray,
    ranks: np.ndarray,
    n_cells: int,
    b: int,
) -> Tuple[np.ndarray, np.ndarray]:
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    for u in range(n_cells):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        rr = ranks[a:z]
        if nb.size <= b:
            chosen = nb
        else:
            pos = np.argpartition(rr, b - 1)[:b]
            chosen = nb[pos]
        if chosen.size:
            src_parts.append(np.full(chosen.size, u, dtype=np.int32))
            dst_parts.append(chosen.astype(np.int32, copy=False))
    if not src_parts:
        return np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    keys = src.astype(np.int64) * n_cells + dst.astype(np.int64)
    rev = dst.astype(np.int64) * n_cells + src.astype(np.int64)
    out = keys_to_csr(np.concatenate((keys, rev)), n_cells)
    log(f"uniform bottom-{b}: edges={out[1].size:,}, degree={out[1].size / n_cells:.2f}")
    return out


def geometric_core_graph(
    xb: np.ndarray,
    portals: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    g: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_cells = len(portals)
    src = edge_sources(indptr)
    dist = np.empty(indices.size, dtype=np.float32)
    t0 = time.perf_counter()
    chunk = 200_000
    for start in range(0, indices.size, chunk):
        end = min(indices.size, start + chunk)
        a = xb[portals[src[start:end]]]
        b = xb[portals[indices[start:end]]]
        diff = a - b
        dist[start:end] = np.einsum("ij,ij->i", diff, diff, optimize=True)
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    for u in range(n_cells):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        dd = dist[a:z]
        if nb.size <= g:
            chosen = nb
        else:
            pos = np.argpartition(dd, g - 1)[:g]
            chosen = nb[pos]
        if chosen.size:
            src_parts.append(np.full(chosen.size, u, dtype=np.int32))
            dst_parts.append(chosen.astype(np.int32, copy=False))
    src2 = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int32)
    dst2 = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int32)
    keys = src2.astype(np.int64) * n_cells + dst2.astype(np.int64)
    out = keys_to_csr(keys, n_cells)
    elapsed = time.perf_counter() - t0
    log(f"geometric core-{g}: directed edges={out[1].size:,}, seconds={elapsed:.2f}")
    return out[0], out[1], elapsed


def filter_active_graph(
    indptr: np.ndarray,
    indices: np.ndarray,
    active: np.ndarray,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    t0 = time.perf_counter()
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    summary_checks = 0
    for u in np.flatnonzero(active):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        summary_checks += int(nb.size)
        chosen = nb[active[nb]]
        if chosen.size:
            src_parts.append(np.full(chosen.size, u, dtype=np.int32))
            dst_parts.append(chosen.astype(np.int32, copy=False))
    if src_parts:
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        keys = src.astype(np.int64) * n_cells + dst.astype(np.int64)
        out = keys_to_csr(keys, n_cells)
    else:
        out = (np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32))
    return out[0], out[1], time.perf_counter() - t0, summary_checks


def materialize_capacity_graph(
    full_indptr: np.ndarray,
    full_indices: np.ndarray,
    neglog: np.ndarray,
    counts: np.ndarray,
    b: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    n_cells = len(counts)
    active = counts > 0
    t0 = time.perf_counter()
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    summary_checks = 0
    for u in np.flatnonzero(active):
        a, z = int(full_indptr[u]), int(full_indptr[u + 1])
        nb = full_indices[a:z]
        summary_checks += int(nb.size)
        keep = active[nb]
        nb2 = nb[keep]
        if nb2.size == 0:
            continue
        score = neglog[a:z][keep] / counts[nb2].astype(np.float64)
        if nb2.size > b:
            pos = np.argpartition(score, b - 1)[:b]
            chosen = nb2[pos]
        else:
            chosen = nb2
        src_parts.append(np.full(chosen.size, u, dtype=np.int32))
        dst_parts.append(chosen.astype(np.int32, copy=False))
    if src_parts:
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        keys = src.astype(np.int64) * n_cells + dst.astype(np.int64)
        rev = dst.astype(np.int64) * n_cells + src.astype(np.int64)
        out = keys_to_csr(np.concatenate((keys, rev)), n_cells)
    else:
        out = (np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32))
    return out[0], out[1], time.perf_counter() - t0, summary_checks


def materialize_hybrid_graph(
    full_indptr: np.ndarray,
    full_indices: np.ndarray,
    ranks: np.ndarray,
    neglog: np.ndarray,
    geo_indptr: np.ndarray,
    geo_indices: np.ndarray,
    counts: np.ndarray,
    cap_b: int,
    uni_b: int,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    n_cells = len(counts)
    active = counts > 0
    t0 = time.perf_counter()
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    summary_checks = 0
    for u in np.flatnonzero(active):
        a, z = int(full_indptr[u]), int(full_indptr[u + 1])
        nb = full_indices[a:z]
        summary_checks += int(nb.size)
        active_mask = active[nb]
        nb_active = nb[active_mask]
        if nb_active.size == 0:
            continue
        chosen_parts: List[np.ndarray] = []
        ga, gz = int(geo_indptr[u]), int(geo_indptr[u + 1])
        core = geo_indices[ga:gz]
        core = core[active[core]]
        if core.size:
            chosen_parts.append(core)
        score = neglog[a:z][active_mask] / counts[nb_active].astype(np.float64)
        if cap_b > 0:
            if nb_active.size > cap_b:
                pos = np.argpartition(score, cap_b - 1)[:cap_b]
                chosen_parts.append(nb_active[pos])
            else:
                chosen_parts.append(nb_active)
        if uni_b > 0:
            rr = ranks[a:z][active_mask]
            if nb_active.size > uni_b:
                pos = np.argpartition(rr, uni_b - 1)[:uni_b]
                chosen_parts.append(nb_active[pos])
            else:
                chosen_parts.append(nb_active)
        chosen = np.unique(np.concatenate(chosen_parts)).astype(np.int32, copy=False)
        src_parts.append(np.full(chosen.size, u, dtype=np.int32))
        dst_parts.append(chosen)
    if src_parts:
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        keys = src.astype(np.int64) * n_cells + dst.astype(np.int64)
        rev = dst.astype(np.int64) * n_cells + src.astype(np.int64)
        out = keys_to_csr(np.concatenate((keys, rev)), n_cells)
    else:
        out = (np.zeros(n_cells + 1, dtype=np.int64), np.empty(0, dtype=np.int32))
    return out[0], out[1], time.perf_counter() - t0, summary_checks


def grouped_ids(cells: np.ndarray, mask: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.flatnonzero(mask).astype(np.int32)
    labels = cells[ids]
    order = np.argsort(labels, kind="stable")
    ids = ids[order]
    labels = labels[order]
    counts = np.bincount(labels, minlength=n_cells).astype(np.int32)
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return ids, offsets, counts


def children_csr(parent: np.ndarray, n_parents: int) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(parent, kind="stable")
    counts = np.bincount(parent, minlength=n_parents)
    indptr = np.empty(n_parents + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, order.astype(np.int32)


def predicate_seed_cells(
    mask: np.ndarray,
    levels_raw: np.ndarray,
    cells: np.ndarray,
    seed: int,
    limit: int = 4,
) -> np.ndarray:
    ids = np.flatnonzero(mask).astype(np.int32)
    if ids.size == 0:
        return np.empty(0, dtype=np.int32)
    level = levels_raw[ids]
    rank = splitmix64_array(ids.astype(np.uint64) ^ np.uint64(seed))
    order = np.lexsort((rank, -level))
    out: List[int] = []
    seen: set[int] = set()
    for pos in order.tolist():
        c = int(cells[ids[pos]])
        if c not in seen:
            seen.add(c)
            out.append(c)
            if len(out) >= limit:
                break
    return np.asarray(out, dtype=np.int32)


@dataclass
class SearchStats:
    answer: np.ndarray
    portal_computations: int
    payload_computations: int
    adjacency_reads: int
    visited_cells: int
    latency_ms: float


def graph_search(
    xb: np.ndarray,
    q: np.ndarray,
    portals: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    seeds: np.ndarray,
    ef: int,
    qualified: np.ndarray,
    qoff: np.ndarray,
    seed_distances: np.ndarray | None = None,
) -> SearchStats:
    t0 = time.perf_counter()
    n_cells = len(portals)
    ef = min(max(1, int(ef)), n_cells)
    seeds = np.unique(seeds[(seeds >= 0) & (seeds < n_cells)]).astype(np.int32)
    if seeds.size == 0:
        return SearchStats(np.empty(0, dtype=np.int32), 0, 0, 0, 0, 1000 * (time.perf_counter() - t0))
    visited = np.zeros(n_cells, dtype=np.uint8)
    if seed_distances is None:
        ds0 = l2_batch(xb[portals[seeds]], q)
        seed_comp_count = int(seeds.size)
    else:
        if len(seed_distances) != len(seeds):
            raise ValueError("seed_distances must align with unique seeds")
        ds0 = np.asarray(seed_distances, dtype=np.float32)
        seed_comp_count = 0
    candidates: List[Tuple[float, int]] = []
    top: List[Tuple[float, int]] = []
    for d0, s0 in zip(ds0.tolist(), seeds.tolist()):
        visited[s0] = 1
        heapq.heappush(candidates, (float(d0), int(s0)))
        heapq.heappush(top, (-float(d0), int(s0)))
        if len(top) > ef:
            heapq.heappop(top)
    portal_comps = seed_comp_count
    adjacency_reads = 0
    while candidates:
        du, u = heapq.heappop(candidates)
        worst = -top[0][0]
        if len(top) >= ef and du > worst:
            break
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        adjacency_reads += int(nb.size)
        if nb.size == 0:
            continue
        unseen = nb[visited[nb] == 0]
        if unseen.size == 0:
            continue
        visited[unseen] = 1
        ds = l2_batch(xb[portals[unseen]], q)
        portal_comps += int(unseen.size)
        for dv, v0 in zip(ds.tolist(), unseen.tolist()):
            v = int(v0)
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(candidates, (float(dv), v))
                heapq.heappush(top, (-float(dv), v))
                if len(top) > ef:
                    heapq.heappop(top)
    selected = [u for _, u in sorted([(-negd, u) for negd, u in top])]
    payload_parts: List[np.ndarray] = []
    for c in selected:
        a, z = int(qoff[c]), int(qoff[c + 1])
        if z > a:
            payload_parts.append(qualified[a:z])
    if payload_parts:
        payload = np.concatenate(payload_parts)
        answer = topk_ids(xb, q, payload, 10)
    else:
        payload = np.empty(0, dtype=np.int32)
        answer = payload
    return SearchStats(
        answer=answer,
        portal_computations=portal_comps,
        payload_computations=int(payload.size),
        adjacency_reads=adjacency_reads,
        visited_cells=int(np.count_nonzero(visited)),
        latency_ms=1000 * (time.perf_counter() - t0),
    )


def descend_topdown_seeds(
    xb: np.ndarray,
    q: np.ndarray,
    level_portals: Mapping[int, np.ndarray],
    level_counts: Mapping[int, np.ndarray],
    child_structures: Mapping[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    target_level: int,
    ef: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, float]:
    t0 = time.perf_counter()
    current_level = 4
    active = np.flatnonzero(level_counts[current_level] > 0).astype(np.int32)
    summary_checks = int(len(level_counts[current_level]))
    if active.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0, summary_checks, 0, 1000 * (time.perf_counter() - t0)
    ds = l2_batch(xb[level_portals[current_level][active]], q)
    portal_comps = int(active.size)
    beam4 = min(active.size, max(4, int(math.ceil(math.sqrt(max(ef, 1))))))
    if active.size > beam4:
        pos = np.argpartition(ds, beam4 - 1)[:beam4]
        frontier = active[pos]
        frontier_ds = ds[pos]
    else:
        frontier = active
        frontier_ds = ds
    child_reads = 0
    for next_level in range(current_level - 1, target_level - 1, -1):
        ip, ix = child_structures[(next_level, current_level)]
        parts: List[np.ndarray] = []
        for p in frontier.tolist():
            a, z = int(ip[p]), int(ip[p + 1])
            child_reads += z - a
            if z > a:
                parts.append(ix[a:z])
        if not parts:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), portal_comps, summary_checks, child_reads, 1000 * (time.perf_counter() - t0)
        children = np.unique(np.concatenate(parts)).astype(np.int32)
        summary_checks += int(children.size)
        children = children[level_counts[next_level][children] > 0]
        if children.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), portal_comps, summary_checks, child_reads, 1000 * (time.perf_counter() - t0)
        ds = l2_batch(xb[level_portals[next_level][children]], q)
        portal_comps += int(children.size)
        if next_level == target_level:
            keep = min(children.size, max(8, ef // 4))
        else:
            keep = min(children.size, max(8, ef // 8))
        if children.size > keep:
            pos = np.argpartition(ds, keep - 1)[:keep]
            frontier = children[pos]
            frontier_ds = ds[pos]
        else:
            frontier = children
            frontier_ds = ds
        current_level = next_level
    return frontier.astype(np.int32), np.asarray(frontier_ds, dtype=np.float32), portal_comps, summary_checks, child_reads, 1000 * (time.perf_counter() - t0)


def estimator_table(levels_raw: np.ndarray, masks_by_s: Mapping[float, Sequence[np.ndarray]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    max_level = int(levels_raw.max()) - 1
    for s, masks in masks_by_s.items():
        for mi, mask in enumerate(masks):
            for level in range(0, max_level + 1):
                sample = levels_raw >= level + 1
                n_l = int(sample.sum())
                x_l = int(np.count_nonzero(mask & sample))
                shat = x_l / n_l if n_l else float("nan")
                rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "level": level,
                        "sample_size": n_l,
                        "matches": x_l,
                        "s_hat": shat,
                        "relative_error": abs(shat - s) / s if n_l else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def append_graph_result(
    rows: List[Dict[str, object]],
    method: str,
    selectivity: float,
    mask_seed: int,
    qid: int,
    parameter: int,
    target_level: int,
    stats: SearchStats,
    truth: np.ndarray,
    valid_count: int,
    materialization_ms: float,
    materialization_summary_checks: int,
    extra_portal_comps: int = 0,
    extra_summary_checks: int = 0,
    extra_child_reads: int = 0,
    extra_latency_ms: float = 0.0,
) -> None:
    rows.append(
        {
            "method": method,
            "selectivity": selectivity,
            "mask_seed": mask_seed,
            "query": qid,
            "parameter": parameter,
            "matched_level": target_level,
            "recall": recall_at_10(stats.answer, truth),
            "distance_computations": extra_portal_comps + stats.portal_computations + stats.payload_computations,
            "portal_computations": extra_portal_comps + stats.portal_computations,
            "payload_computations": stats.payload_computations,
            "adjacency_reads": stats.adjacency_reads,
            "visited_cells": stats.visited_cells,
            "summary_checks_query": extra_summary_checks,
            "child_reads_query": extra_child_reads,
            "latency_ms": extra_latency_ms + stats.latency_ms,
            "valid_count": valid_count,
            "materialization_ms": materialization_ms,
            "materialization_summary_checks": materialization_summary_checks,
        }
    )


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        xb = np.asarray(f["train"], dtype=np.float32)
        xq = np.asarray(f["test"][: args.queries], dtype=np.float32)
    n, d = xb.shape
    selectivities = [float(x) for x in args.selectivities.split(",")]
    efs = [int(x) for x in args.efs.split(",")]
    pools = [int(x) for x in args.postfilter_pools.split(",")]
    log(f"loaded dataset: xb={xb.shape}, queries={len(xq)}, selectivities={selectivities}")

    index, build_seconds = build_hnsw(xb, args.M, args.ef_construction, args.threads)
    levels_raw, adj, base_actual_edges, hnsw_neighbor_slots = extract_layer0(index, n)
    level_df = available_levels(levels_raw)
    level_df.to_csv(out / "hnsw_levels.csv", index=False)

    levels_needed = [2, 3, 4]
    level_portals: Dict[int, np.ndarray] = {ell: portal_ids(levels_raw, ell) for ell in levels_needed}
    cell_maps: Dict[int, np.ndarray] = {}
    assignment_rows: List[Dict[str, object]] = []
    quotient: Dict[int, Dict[str, object]] = {}

    for ell in (2, 3):
        cells, seconds, stats = assign_geometric_cells(xb, level_portals[ell], args.threads)
        cell_maps[ell] = cells
        ip, ix, contract_seconds = contract_graph(adj, cells, len(level_portals[ell]))
        ranks, neglog = edge_randoms(ip, ix, args.seed + 10_000 * ell)
        ub8 = uniform_bottomk_graph(ip, ix, ranks, len(level_portals[ell]), args.bottom_b)
        gip, gix, geo_seconds = geometric_core_graph(
            xb, level_portals[ell], ip, ix, args.geo_core
        )
        quotient[ell] = {
            "full": (ip, ix),
            "ranks": ranks,
            "neglog": neglog,
            "uniform": ub8,
            "geo": (gip, gix),
        }
        assignment_rows.append(
            {
                "level": ell,
                "n_cells": len(level_portals[ell]),
                "assignment_seconds": seconds,
                "contract_seconds": contract_seconds,
                "geo_core_seconds": geo_seconds,
                "full_edges": int(ix.size),
                "uniform_edges": int(ub8[1].size),
                **{f"cell_{k}": v for k, v in stats.items()},
            }
        )

    parent23, parent23_seconds = assign_portal_parents(
        xb, level_portals[2], level_portals[3], args.threads
    )
    parent34, parent34_seconds = assign_portal_parents(
        xb, level_portals[3], level_portals[4], args.threads
    )
    child23 = children_csr(parent23, len(level_portals[3]))
    child34 = children_csr(parent34, len(level_portals[4]))
    child_structures = {(2, 3): child23, (3, 4): child34}

    masks_by_s: Dict[float, List[np.ndarray]] = {
        s: [independent_mask(n, s, args.seed + int(round(s * 1e9)) + 1000 * i) for i in range(args.mask_seeds)]
        for s in selectivities
    }
    estimator_table(levels_raw, masks_by_s).to_csv(out / "selectivity_estimator.csv", index=False)

    index.hnsw.efSearch = 64
    _, entry_ids = index.search(xq, 1)
    entry_ids = entry_ids[:, 0].astype(np.int32)

    max_pool = max(pools)
    index.hnsw.efSearch = max_pool
    t0 = time.perf_counter()
    _, post_candidates = index.search(xq, max_pool)
    postfilter_ms_per_query = 1000 * (time.perf_counter() - t0) / len(xq)
    log(f"post-filter candidate pool {max_pool} produced in {postfilter_ms_per_query:.2f} ms/query")

    rows: List[Dict[str, object]] = []
    materialization_rows: List[Dict[str, object]] = []
    structural_rows: List[Dict[str, object]] = []
    truths: Dict[Tuple[float, int, int], np.ndarray] = {}

    qids_by_mask: Dict[int, List[int]] = {
        mi: [qid for qid in range(len(xq)) if qid % args.mask_seeds == mi]
        for mi in range(args.mask_seeds)
    }

    for s in selectivities:
        target = choose_target_level(level_df, s, allowed=(2, 3))
        portals = level_portals[target]
        cells = cell_maps[target]
        full_ip, full_ix = quotient[target]["full"]
        ranks = quotient[target]["ranks"]
        neglog = quotient[target]["neglog"]
        uniform_ip, uniform_ix = quotient[target]["uniform"]
        geo_ip, geo_ix = quotient[target]["geo"]
        log(f"selectivity={s:g}: target=L{target}, cells={len(portals):,}")

        for mi, mask in enumerate(masks_by_s[s]):
            qualified, qoff, counts = grouped_ids(cells, mask, len(portals))
            active = counts > 0
            valid_count = int(qualified.size)

            if target == 2:
                counts3 = np.bincount(parent23, weights=counts, minlength=len(level_portals[3])).astype(np.int32)
                counts4 = np.bincount(parent34, weights=counts3, minlength=len(level_portals[4])).astype(np.int32)
                level_counts = {2: counts, 3: counts3, 4: counts4}
            else:
                counts4 = np.bincount(parent34, weights=counts, minlength=len(level_portals[4])).astype(np.int32)
                level_counts = {3: counts, 4: counts4}

            aip, aix, active_uniform_s, active_uniform_checks = filter_active_graph(
                uniform_ip, uniform_ix, active, len(portals)
            )
            cip, cix, cap_s, cap_checks = materialize_capacity_graph(
                full_ip, full_ix, neglog, counts, args.bottom_b
            )
            hip, hix, hybrid_s, hybrid_checks = materialize_hybrid_graph(
                full_ip,
                full_ix,
                ranks,
                neglog,
                geo_ip,
                geo_ix,
                counts,
                args.cap_b,
                args.uni_b,
            )
            graphs = {
                "uniform_b8_active": (aip, aix, active_uniform_s, active_uniform_checks),
                "capacity_b8": (cip, cix, cap_s, cap_checks),
                "hybrid_g4_c4_u4": (hip, hix, hybrid_s, hybrid_checks),
            }
            for name, (_, gx, sec, checks) in graphs.items():
                materialization_rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "matched_level": target,
                        "method": name,
                        "active_cells": int(active.sum()),
                        "edges": int(gx.size),
                        "mean_degree_active": gx.size / max(int(active.sum()), 1),
                        "materialization_ms": 1000 * sec,
                        "summary_checks": checks,
                    }
                )

            valid_ids = np.flatnonzero(mask).astype(np.int32)
            block = adj[valid_ids]
            safe = np.maximum(block, 0)
            point_deg = np.sum((block >= 0) & mask[safe], axis=1)
            for name, (gip, gix, _, _) in graphs.items():
                degrees: List[int] = []
                for u in np.flatnonzero(active).tolist():
                    nb = gix[int(gip[u]) : int(gip[u + 1])]
                    degrees.append(int(np.count_nonzero(active[nb])))
                arr = np.asarray(degrees, dtype=np.int32)
                structural_rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "matched_level": target,
                        "method": name,
                        "valid_count": valid_count,
                        "active_cells": int(active.sum()),
                        "point_valid_degree": float(point_deg.mean()),
                        "point_zero_degree": float(np.mean(point_deg == 0)),
                        "active_degree": float(arr.mean()) if arr.size else 0.0,
                        "active_zero_degree": float(np.mean(arr == 0)) if arr.size else 1.0,
                    }
                )

            for qid in qids_by_mask[mi]:
                truth = topk_ids(xb, xq[qid], qualified, 10)
                truths[(s, mi, qid)] = truth
                tpre = time.perf_counter()
                _ = topk_ids(xb, xq[qid], qualified, 10)
                rows.append(
                    {
                        "method": "prefilter_exact",
                        "selectivity": s,
                        "mask_seed": mi,
                        "query": qid,
                        "parameter": 0,
                        "matched_level": target,
                        "recall": 1.0,
                        "distance_computations": valid_count,
                        "portal_computations": 0,
                        "payload_computations": valid_count,
                        "adjacency_reads": 0,
                        "visited_cells": 0,
                        "summary_checks_query": 0,
                        "child_reads_query": 0,
                        "latency_ms": 1000 * (time.perf_counter() - tpre),
                        "valid_count": valid_count,
                        "materialization_ms": 0.0,
                        "materialization_summary_checks": 0,
                    }
                )
                for pool in pools:
                    cand = post_candidates[qid, :pool]
                    found = cand[mask[cand]][:10].astype(np.int32)
                    rows.append(
                        {
                            "method": "postfilter_hnsw",
                            "selectivity": s,
                            "mask_seed": mi,
                            "query": qid,
                            "parameter": pool,
                            "matched_level": target,
                            "recall": recall_at_10(found, truth),
                            "distance_computations": np.nan,
                            "portal_computations": np.nan,
                            "payload_computations": int(found.size),
                            "adjacency_reads": np.nan,
                            "visited_cells": np.nan,
                            "summary_checks_query": 0,
                            "child_reads_query": 0,
                            "latency_ms": postfilter_ms_per_query,
                            "valid_count": valid_count,
                            "materialization_ms": 0.0,
                            "materialization_summary_checks": 0,
                        }
                    )

            fixed_entries = cells[entry_ids]
            pseed = predicate_seed_cells(mask, levels_raw, cells, args.seed + mi, limit=args.predicate_seeds)
            for ef in efs:
                for qid in qids_by_mask[mi]:
                    truth = truths[(s, mi, qid)]
                    st = graph_search(
                        xb,
                        xq[qid],
                        portals,
                        uniform_ip,
                        uniform_ix,
                        np.asarray([fixed_entries[qid]], dtype=np.int32),
                        ef,
                        qualified,
                        qoff,
                    )
                    append_graph_result(
                        rows,
                        "uniform_b8_router",
                        s,
                        mi,
                        qid,
                        ef,
                        target,
                        st,
                        truth,
                        valid_count,
                        0.0,
                        0,
                    )

                    for name, (gip, gix, sec, checks) in graphs.items():
                        seeds = pseed
                        if active[fixed_entries[qid]]:
                            seeds = np.unique(np.concatenate((seeds, [fixed_entries[qid]]))).astype(np.int32)
                        st = graph_search(
                            xb,
                            xq[qid],
                            portals,
                            gip,
                            gix,
                            seeds,
                            ef,
                            qualified,
                            qoff,
                        )
                        append_graph_result(
                            rows,
                            name,
                            s,
                            mi,
                            qid,
                            ef,
                            target,
                            st,
                            truth,
                            valid_count,
                            1000 * sec,
                            checks,
                        )

                    td_seeds, td_seed_ds, td_pc, td_checks, td_child_reads, td_ms = descend_topdown_seeds(
                        xb,
                        xq[qid],
                        level_portals,
                        level_counts,
                        child_structures,
                        target,
                        ef,
                    )
                    if td_seeds.size == 0:
                        td_seeds = pseed
                        td_seed_ds = None
                    st = graph_search(
                        xb,
                        xq[qid],
                        portals,
                        hip,
                        hix,
                        td_seeds,
                        ef,
                        qualified,
                        qoff,
                        seed_distances=td_seed_ds,
                    )
                    append_graph_result(
                        rows,
                        "topdown_summary_hybrid",
                        s,
                        mi,
                        qid,
                        ef,
                        target,
                        st,
                        truth,
                        valid_count,
                        1000 * hybrid_s,
                        hybrid_checks,
                        extra_portal_comps=td_pc,
                        extra_summary_checks=td_checks,
                        extra_child_reads=td_child_reads,
                        extra_latency_ms=td_ms,
                    )
                log(f"s={s:g}, mask={mi}, ef={ef} complete")

    query_df = pd.DataFrame(rows)
    query_df.to_csv(out / "query_results.csv", index=False)
    summary_df = (
        query_df.groupby(["method", "selectivity", "parameter", "matched_level"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            recall_p10=("recall", lambda x: float(np.quantile(x, 0.10))),
            distance_computations=("distance_computations", "mean"),
            portal_computations=("portal_computations", "mean"),
            payload_computations=("payload_computations", "mean"),
            adjacency_reads=("adjacency_reads", "mean"),
            visited_cells=("visited_cells", "mean"),
            summary_checks_query=("summary_checks_query", "mean"),
            child_reads_query=("child_reads_query", "mean"),
            latency_ms=("latency_ms", "mean"),
            valid_count=("valid_count", "mean"),
            materialization_ms=("materialization_ms", "mean"),
            materialization_summary_checks=("materialization_summary_checks", "mean"),
        )
        .sort_values(["selectivity", "method", "parameter"])
    )
    summary_df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(materialization_rows).to_csv(out / "materialization.csv", index=False)
    pd.DataFrame(structural_rows).to_csv(out / "structural.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(out / "index_levels.csv", index=False)

    metadata = {
        "dataset": str(args.dataset),
        "n": n,
        "d": d,
        "queries": len(xq),
        "M": args.M,
        "ef_construction": args.ef_construction,
        "threads": args.threads,
        "selectivities": selectivities,
        "efs": efs,
        "mask_seeds": args.mask_seeds,
        "bottom_b": args.bottom_b,
        "geo_core": args.geo_core,
        "cap_b": args.cap_b,
        "uni_b": args.uni_b,
        "build_seconds": build_seconds,
        "base_actual_edges": base_actual_edges,
        "hnsw_neighbor_slots": hnsw_neighbor_slots,
        "parent23_seconds": parent23_seconds,
        "parent34_seconds": parent34_seconds,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"complete: query rows={len(query_df):,}, out={out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/sift-128-euclidean.hdf5")
    p.add_argument("--out", default="results/rankq_summary")
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--ef-construction", type=int, default=80)
    p.add_argument("--mask-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--selectivities", default="0.02,0.01,0.007,0.005,0.003,0.002,0.001")
    p.add_argument("--efs", default="32,64,128,256")
    p.add_argument("--postfilter-pools", default="500,1000,2000,4000,8000,16000")
    p.add_argument("--bottom-b", type=int, default=8)
    p.add_argument("--geo-core", type=int, default=4)
    p.add_argument("--cap-b", type=int, default=4)
    p.add_argument("--uni-b", type=int, default=4)
    p.add_argument("--predicate-seeds", type=int, default=4)
    run(p.parse_args())

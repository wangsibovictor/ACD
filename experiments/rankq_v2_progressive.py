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

import sift1m_1pct_experiment as rq


@dataclass
class CellSearchResult:
    selected_cells: np.ndarray
    portal_computations: int
    adjacency_reads: int
    visited_cells: int
    summary_checks: int
    latency_ms: float


@dataclass
class LevelIndex:
    name: str
    portals: np.ndarray
    cells: np.ndarray
    full_indptr: np.ndarray
    full_indices: np.ndarray
    copies: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray]]]
    cell_stats: Dict[str, float]
    assignment_seconds: float
    contract_seconds: float


@dataclass
class WitnessIndex:
    upper_portals: np.ndarray
    parent: np.ndarray
    child_indptr: np.ndarray
    child_indices: np.ndarray
    upper_indptr: np.ndarray
    upper_indices: np.ndarray
    edge_keys: np.ndarray
    witness_indptr: np.ndarray
    witness_children: np.ndarray


def log(msg: str) -> None:
    rq.log(msg)


def csr_union(graphs: Sequence[Tuple[np.ndarray, np.ndarray]], n: int) -> Tuple[np.ndarray, np.ndarray]:
    if not graphs:
        return np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
    keys_parts: List[np.ndarray] = []
    for indptr, indices in graphs:
        src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        if indices.size:
            keys_parts.append(src * n + indices.astype(np.int64, copy=False))
    if not keys_parts:
        return np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int32)
    return rq.keys_to_csr(np.concatenate(keys_parts), n)


def filter_active_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    active: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(active)
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    checks = 0
    for u in np.flatnonzero(active).tolist():
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        checks += int(nb.size)
        keep = nb[active[nb]]
        if keep.size:
            src_parts.append(np.full(keep.size, u, dtype=np.int32))
            dst_parts.append(keep.astype(np.int32, copy=False))
    if not src_parts:
        return np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int32), checks
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    keys = src.astype(np.int64) * n + dst.astype(np.int64)
    ip, ix = rq.keys_to_csr(keys, n)
    return ip, ix, checks


def certified_router_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    active: np.ndarray,
    min_active_fanout: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """One-hop router recovery: active->active/router and router->active only."""
    n = len(active)
    active_fanout = np.zeros(n, dtype=np.int32)
    checks = 0
    for u in range(n):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        checks += int(nb.size)
        active_fanout[u] = int(np.count_nonzero(active[nb]))
    router = (~active) & (active_fanout >= min_active_fanout)
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    for u in range(n):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        if active[u]:
            keep = nb[active[nb] | router[nb]]
        elif router[u]:
            keep = nb[active[nb]]
        else:
            continue
        if keep.size:
            src_parts.append(np.full(keep.size, u, dtype=np.int32))
            dst_parts.append(keep.astype(np.int32, copy=False))
    if not src_parts:
        return np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int32), checks, int(router.sum())
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    keys = src.astype(np.int64) * n + dst.astype(np.int64)
    ip, ix = rq.keys_to_csr(keys, n)
    return ip, ix, checks, int(router.sum())


def search_cells(
    xb: np.ndarray,
    q: np.ndarray,
    portals: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    seeds: np.ndarray,
    ef: int,
) -> CellSearchResult:
    t0 = time.perf_counter()
    n = len(portals)
    ef = min(max(1, int(ef)), n)
    seeds = np.unique(seeds[(seeds >= 0) & (seeds < n)]).astype(np.int32)
    if seeds.size == 0:
        return CellSearchResult(np.empty(0, dtype=np.int32), 0, 0, 0, 0, 0.0)
    visited = np.zeros(n, dtype=np.uint8)
    ds0 = rq.l2_batch(xb[portals[seeds]], q)
    candidates: List[Tuple[float, int]] = []
    top: List[Tuple[float, int]] = []
    for dv, s0 in zip(ds0.tolist(), seeds.tolist()):
        visited[s0] = 1
        heapq.heappush(candidates, (float(dv), int(s0)))
        heapq.heappush(top, (-float(dv), int(s0)))
        if len(top) > ef:
            heapq.heappop(top)
    portal_comps = int(seeds.size)
    reads = 0
    while candidates:
        du, u = heapq.heappop(candidates)
        if len(top) >= ef and du > -top[0][0]:
            break
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        reads += int(nb.size)
        if nb.size == 0:
            continue
        unseen = nb[visited[nb] == 0]
        if unseen.size == 0:
            continue
        visited[unseen] = 1
        ds = rq.l2_batch(xb[portals[unseen]], q)
        portal_comps += int(unseen.size)
        for dv, v0 in zip(ds.tolist(), unseen.tolist()):
            v = int(v0)
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(candidates, (float(dv), v))
                heapq.heappush(top, (-float(dv), v))
                if len(top) > ef:
                    heapq.heappop(top)
    selected = np.asarray([u for _, u in sorted([(-d, u) for d, u in top])], dtype=np.int32)
    return CellSearchResult(
        selected_cells=selected,
        portal_computations=portal_comps,
        adjacency_reads=reads,
        visited_cells=int(np.count_nonzero(visited)),
        summary_checks=0,
        latency_ms=1000.0 * (time.perf_counter() - t0),
    )


def lazy_degree_refine_search(
    xb: np.ndarray,
    q: np.ndarray,
    portals: np.ndarray,
    base_indptr: np.ndarray,
    base_indices: np.ndarray,
    full_indptr: np.ndarray,
    full_indices: np.ndarray,
    counts: np.ndarray,
    seeds: np.ndarray,
    ef: int,
    degree_target: int,
    max_extra: int,
    capacity_lambda: float,
) -> CellSearchResult:
    """Active-first search; low-degree expansions lazily add query-nearest full-boundary exits."""
    t0 = time.perf_counter()
    active = counts > 0
    n = len(portals)
    ef = min(max(1, int(ef)), n)
    seeds = np.unique(seeds[(seeds >= 0) & (seeds < n) & active[seeds]]).astype(np.int32)
    if seeds.size == 0:
        return CellSearchResult(np.empty(0, dtype=np.int32), 0, 0, 0, 0, 0.0)
    visited = np.zeros(n, dtype=np.uint8)
    ds0 = rq.l2_batch(xb[portals[seeds]], q)
    candidates: List[Tuple[float, int]] = []
    top: List[Tuple[float, int]] = []
    for dv, s0 in zip(ds0.tolist(), seeds.tolist()):
        visited[s0] = 1
        heapq.heappush(candidates, (float(dv), int(s0)))
        heapq.heappush(top, (-float(dv), int(s0)))
        if len(top) > ef:
            heapq.heappop(top)
    portal_comps = int(seeds.size)
    reads = 0
    checks = 0
    while candidates:
        du, u = heapq.heappop(candidates)
        if len(top) >= ef and du > -top[0][0]:
            break
        a, z = int(base_indptr[u]), int(base_indptr[u + 1])
        base_nb = base_indices[a:z]
        reads += int(base_nb.size)
        if "distance_cache" not in locals():
            distance_cache = np.full(n, np.nan, dtype=np.float32)
            distance_cache[seeds] = ds0.astype(np.float32, copy=False)
        chosen = base_nb[active[base_nb]]
        checks += int(base_nb.size)
        if chosen.size < degree_target:
            fa, fz = int(full_indptr[u]), int(full_indptr[u + 1])
            full_nb = full_indices[fa:fz]
            reads += int(full_nb.size)
            checks += int(full_nb.size)
            cand = full_nb[active[full_nb]]
            if cand.size:
                if chosen.size:
                    cand = cand[~np.isin(cand, chosen, assume_unique=False)]
                if cand.size:
                    cand_ds = rq.l2_batch(xb[portals[cand]], q)
                    portal_comps += int(cand.size)
                    weight = 1.0 + capacity_lambda * np.log1p(counts[cand].astype(np.float64))
                    score = cand_ds / weight
                    need = min(max_extra, max(0, degree_target - int(chosen.size)))
                    if cand.size > need > 0:
                        pos = np.argpartition(score, need - 1)[:need]
                        extra = cand[pos]
                        extra_ds = cand_ds[pos]
                    else:
                        extra = cand[:need]
                        extra_ds = cand_ds[:need]
                    if extra.size:
                        distance_cache[extra] = extra_ds
                        chosen = np.unique(np.concatenate((chosen, extra))).astype(np.int32)
        if chosen.size == 0:
            continue
        unseen = chosen[visited[chosen] == 0]
        if unseen.size == 0:
            continue
        visited[unseen] = 1
        cached = distance_cache[unseen]
        miss = np.isnan(cached)
        if np.any(miss):
            ds_new = rq.l2_batch(xb[portals[unseen[miss]]], q)
            portal_comps += int(np.count_nonzero(miss))
            cached[miss] = ds_new
            distance_cache[unseen[miss]] = ds_new
        for dv, v0 in zip(cached.tolist(), unseen.tolist()):
            v = int(v0)
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(candidates, (float(dv), v))
                heapq.heappush(top, (-float(dv), v))
                if len(top) > ef:
                    heapq.heappop(top)
    selected = np.asarray([u for _, u in sorted([(-d, u) for d, u in top])], dtype=np.int32)
    return CellSearchResult(
        selected_cells=selected,
        portal_computations=portal_comps,
        adjacency_reads=reads,
        visited_cells=int(np.count_nonzero(visited)),
        summary_checks=checks,
        latency_ms=1000.0 * (time.perf_counter() - t0),
    )


def adaptive_copy_graph(
    copy_graphs: Sequence[Tuple[np.ndarray, np.ndarray]],
    active: np.ndarray,
    degree_target: int,
    max_copies: int,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """Add independent B8 copies until each active source has enough active exits."""
    n = len(active)
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    checks = 0
    copies_used: List[int] = []
    active_degrees: List[int] = []
    for u in np.flatnonzero(active).tolist():
        acc: List[np.ndarray] = []
        unique = np.empty(0, dtype=np.int32)
        used = 0
        for ip, ix in copy_graphs[:max_copies]:
            a, z = int(ip[u]), int(ip[u + 1])
            nb = ix[a:z]
            checks += int(nb.size)
            good = nb[active[nb]]
            if good.size:
                acc.append(good)
                unique = np.unique(np.concatenate(acc)).astype(np.int32)
            used += 1
            if unique.size >= degree_target:
                break
        copies_used.append(used)
        active_degrees.append(int(unique.size))
        if unique.size:
            src_parts.append(np.full(unique.size, u, dtype=np.int32))
            dst_parts.append(unique)
    if not src_parts:
        return (
            np.zeros(n + 1, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            checks,
            float(np.mean(copies_used)) if copies_used else 0.0,
            0.0,
        )
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    keys = src.astype(np.int64) * n + dst.astype(np.int64)
    ip, ix = rq.keys_to_csr(keys, n)
    return (
        ip,
        ix,
        checks,
        float(np.mean(copies_used)),
        float(np.mean(active_degrees)),
    )


def make_virtual_portals(
    levels_raw: np.ndarray,
    low_level: int,
    high_level: int,
    target_rate: float,
    seed: int,
) -> np.ndarray:
    low = rq.portal_ids(levels_raw, low_level)
    high = rq.portal_ids(levels_raw, high_level)
    n = len(levels_raw)
    target_n = max(len(high), min(len(low), int(round(target_rate * n))))
    high_mask = np.zeros(n, dtype=bool)
    high_mask[high] = True
    candidates = low[~high_mask[low]]
    need = target_n - len(high)
    if need <= 0:
        return np.sort(high).astype(np.int32)
    ranks = rq.splitmix64_array(candidates.astype(np.uint64) ^ np.uint64(seed))
    if need < len(candidates):
        pos = np.argpartition(ranks, need - 1)[:need]
        chosen = candidates[pos]
    else:
        chosen = candidates
    return np.sort(np.concatenate((high, chosen))).astype(np.int32)


def build_level_index(
    name: str,
    xb: np.ndarray,
    adj: np.ndarray,
    portals: np.ndarray,
    threads: int,
    b: int,
    copies: int,
    seed: int,
) -> LevelIndex:
    cells, assignment_seconds, cell_stats = rq.assign_geometric_cells(xb, portals, threads)
    full_ip, full_ix, contract_seconds = rq.contract_graph(adj, cells, len(portals))
    copy_list: List[Tuple[np.ndarray, np.ndarray]] = []
    for ci in range(copies):
        ranks, _ = rq.edge_randoms(full_ip, full_ix, seed + 1_000_003 * ci)
        copy_list.append(rq.uniform_bottomk_graph(full_ip, full_ix, ranks, len(portals), b))
    return LevelIndex(
        name=name,
        portals=portals,
        cells=cells,
        full_indptr=full_ip,
        full_indices=full_ix,
        copies={(b, 0): copy_list},
        cell_stats=cell_stats,
        assignment_seconds=assignment_seconds,
        contract_seconds=contract_seconds,
    )


def build_witness_index(
    lower: LevelIndex,
    upper_portals: np.ndarray,
    xb: np.ndarray,
    threads: int,
) -> Tuple[WitnessIndex, float]:
    parent, parent_seconds = rq.assign_portal_parents(xb, lower.portals, upper_portals, threads)
    child_ip, child_ix = rq.children_csr(parent, len(upper_portals))
    src = rq.edge_sources(lower.full_indptr)
    dst = lower.full_indices
    up_src = parent[src]
    up_dst = parent[dst]
    cross = up_src != up_dst
    up_keys = up_src[cross].astype(np.int64) * len(upper_portals) + up_dst[cross].astype(np.int64)
    edge_keys = np.unique(up_keys)
    upper_ip, upper_ix = rq.keys_to_csr(edge_keys, len(upper_portals))
    witness_code = up_keys.astype(np.int64) * len(lower.portals) + dst[cross].astype(np.int64)
    witness_code = np.unique(witness_code)
    witness_edge_key = witness_code // len(lower.portals)
    witness_child = (witness_code % len(lower.portals)).astype(np.int32)
    edge_pos = np.searchsorted(edge_keys, witness_edge_key)
    order = np.argsort(edge_pos, kind="stable")
    edge_pos = edge_pos[order]
    witness_child = witness_child[order]
    counts = np.bincount(edge_pos, minlength=len(edge_keys))
    witness_ip = np.empty(len(edge_keys) + 1, dtype=np.int64)
    witness_ip[0] = 0
    np.cumsum(counts, out=witness_ip[1:])
    return (
        WitnessIndex(
            upper_portals=upper_portals,
            parent=parent,
            child_indptr=child_ip,
            child_indices=child_ix,
            upper_indptr=upper_ip,
            upper_indices=upper_ix,
            edge_keys=edge_keys,
            witness_indptr=witness_ip,
            witness_children=witness_child,
        ),
        parent_seconds,
    )


def witness_descent_seeds(
    xb: np.ndarray,
    q: np.ndarray,
    lower: LevelIndex,
    witness: WitnessIndex,
    upper_graph: Tuple[np.ndarray, np.ndarray],
    selected_upper: np.ndarray,
    counts_lower: np.ndarray,
    seed_budget: int,
) -> Tuple[np.ndarray, int, int, int, float]:
    t0 = time.perf_counter()
    n_upper = len(witness.upper_portals)
    candidates: List[np.ndarray] = []
    child_reads = 0
    summary_checks = 0
    for u in selected_upper.tolist():
        # One local active child protects the current coarse region.
        ca, cz = int(witness.child_indptr[u]), int(witness.child_indptr[u + 1])
        local = witness.child_indices[ca:cz]
        child_reads += int(local.size)
        summary_checks += int(local.size)
        local = local[counts_lower[local] > 0]
        if local.size:
            candidates.append(local)
        # Edge-specific witness destinations protect shortcut landing regions.
        a, z = int(upper_graph[0][u]), int(upper_graph[0][u + 1])
        for v in upper_graph[1][a:z].tolist():
            key = int(u) * n_upper + int(v)
            ep = int(np.searchsorted(witness.edge_keys, key))
            if ep >= len(witness.edge_keys) or int(witness.edge_keys[ep]) != key:
                continue
            wa, wz = int(witness.witness_indptr[ep]), int(witness.witness_indptr[ep + 1])
            ch = witness.witness_children[wa:wz]
            child_reads += int(ch.size)
            summary_checks += int(ch.size)
            ch = ch[counts_lower[ch] > 0]
            if ch.size:
                candidates.append(ch)
    if not candidates:
        return np.empty(0, dtype=np.int32), 0, summary_checks, child_reads, 0.0
    cand = np.unique(np.concatenate(candidates)).astype(np.int32)
    ds = rq.l2_batch(xb[lower.portals[cand]], q)
    comps = int(cand.size)
    keep = min(seed_budget, cand.size)
    if cand.size > keep:
        pos = np.argpartition(ds, keep - 1)[:keep]
        cand = cand[pos]
    return cand, comps, summary_checks, child_reads, 1000.0 * (time.perf_counter() - t0)


def payload_answer(
    xb: np.ndarray,
    q: np.ndarray,
    qualified: np.ndarray,
    qoff: np.ndarray,
    selected_cells: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    parts: List[np.ndarray] = []
    for c in selected_cells.tolist():
        a, z = int(qoff[c]), int(qoff[c + 1])
        if z > a:
            parts.append(qualified[a:z])
    if not parts:
        return np.empty(0, dtype=np.int32), 0, 1000.0 * (time.perf_counter() - t0)
    payload = np.concatenate(parts)
    answer = rq.topk_ids(xb, q, payload, 10)
    return answer, int(payload.size), 1000.0 * (time.perf_counter() - t0)


def append_result(
    rows: List[Dict[str, object]],
    method: str,
    selectivity: float,
    mask_seed: int,
    qid: int,
    parameter: int,
    level_name: str,
    search: CellSearchResult,
    answer: np.ndarray,
    truth: np.ndarray,
    payload_computations: int,
    payload_ms: float,
    valid_count: int,
    materialization_ms: float,
    materialization_checks: int,
    extra_portal_computations: int = 0,
    extra_adjacency_reads: int = 0,
    extra_summary_checks: int = 0,
    extra_child_reads: int = 0,
    extra_latency_ms: float = 0.0,
    copies_used_mean: float = 0.0,
    active_degree_mean: float = 0.0,
) -> None:
    rows.append(
        {
            "method": method,
            "selectivity": selectivity,
            "mask_seed": mask_seed,
            "query": qid,
            "parameter": parameter,
            "level_name": level_name,
            "recall": rq.recall_at_10(answer, truth),
            "distance_computations": extra_portal_computations + search.portal_computations + payload_computations,
            "portal_computations": extra_portal_computations + search.portal_computations,
            "payload_computations": payload_computations,
            "adjacency_reads": extra_adjacency_reads + search.adjacency_reads,
            "visited_cells": search.visited_cells,
            "summary_checks_query": extra_summary_checks + search.summary_checks,
            "child_reads_query": extra_child_reads,
            "latency_ms": extra_latency_ms + search.latency_ms + payload_ms,
            "valid_count": valid_count,
            "materialization_ms": materialization_ms,
            "materialization_summary_checks": materialization_checks,
            "copies_used_mean": copies_used_mean,
            "active_degree_mean": active_degree_mean,
        }
    )


def choose_level_index(levels: Mapping[str, LevelIndex], s: float) -> LevelIndex:
    scored = []
    for name, idx in levels.items():
        rate = len(idx.portals) / len(idx.cells)
        scored.append((abs(math.log(rate / s)), name, idx))
    scored.sort(key=lambda x: x[0])
    return scored[0][2]


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        xb = np.asarray(f["train"], dtype=np.float32)
        xq = np.asarray(f["test"][: args.queries], dtype=np.float32)
    n, d = xb.shape
    selectivities = [float(x) for x in args.selectivities.split(",")]
    efs = [int(x) for x in args.efs.split(",")]
    post_pools = [int(x) for x in args.postfilter_pools.split(",")]
    log(f"loaded dataset xb={xb.shape}, queries={len(xq)}, s={selectivities}")

    index, build_seconds = rq.build_hnsw(xb, args.M, args.ef_construction, args.threads)
    levels_raw, adj, base_actual_edges, hnsw_neighbor_slots = rq.extract_layer0(index, n)

    l2 = rq.portal_ids(levels_raw, 2)
    l3 = rq.portal_ids(levels_raw, 3)
    l4 = rq.portal_ids(levels_raw, 4)
    v08 = make_virtual_portals(levels_raw, 2, 3, args.virtual_rate_hi, args.seed + 8101)
    v04 = make_virtual_portals(levels_raw, 2, 3, args.virtual_rate_lo, args.seed + 4101)
    portal_sets = {
        "L2": l2,
        "V08": v08,
        "V04": v04,
        "L3": l3,
    }

    level_indices: Dict[str, LevelIndex] = {}
    index_rows: List[Dict[str, object]] = []
    for li, (name, portals) in enumerate(portal_sets.items()):
        log(f"building level {name}: portals={len(portals):,}, rate={len(portals)/n:.6f}")
        idx = build_level_index(
            name,
            xb,
            adj,
            portals,
            args.threads,
            args.bottom_b,
            args.max_copies,
            args.seed + 100_000 * li,
        )
        level_indices[name] = idx
        copies = idx.copies[(args.bottom_b, 0)]
        index_rows.append(
            {
                "level_name": name,
                "n_cells": len(portals),
                "sampling_rate": len(portals) / n,
                "mean_cell_size": n / len(portals),
                "assignment_seconds": idx.assignment_seconds,
                "contract_seconds": idx.contract_seconds,
                "full_edges": int(idx.full_indices.size),
                "copy1_edges": int(copies[0][1].size),
                "copy4_union_edges": int(csr_union(copies[: min(4, len(copies))], len(portals))[1].size),
                "copy8_union_edges": int(csr_union(copies[: min(8, len(copies))], len(portals))[1].size),
                **{f"cell_{k}": v for k, v in idx.cell_stats.items()},
            }
        )

    witnesses: Dict[str, WitnessIndex] = {}
    witness_upper_copy: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    witness_rows: List[Dict[str, object]] = []
    for wi, (name, idx) in enumerate(level_indices.items()):
        upper_portals = l4 if name == "L3" else l3
        witness, parent_seconds = build_witness_index(idx, upper_portals, xb, args.threads)
        ranks, _ = rq.edge_randoms(
            witness.upper_indptr,
            witness.upper_indices,
            args.seed + 700_000 + wi,
        )
        upper_copy = rq.uniform_bottomk_graph(
            witness.upper_indptr,
            witness.upper_indices,
            ranks,
            len(upper_portals),
            args.bottom_b,
        )
        witnesses[name] = witness
        witness_upper_copy[name] = upper_copy
        witness_rows.append(
            {
                "level_name": name,
                "upper_cells": len(upper_portals),
                "parent_seconds": parent_seconds,
                "upper_full_edges": int(witness.upper_indices.size),
                "upper_copy_edges": int(upper_copy[1].size),
                "witness_children": int(witness.witness_children.size),
                "mean_witness_children_per_upper_edge": witness.witness_children.size / max(witness.upper_indices.size, 1),
            }
        )

    pd.DataFrame(index_rows).to_csv(out / "index_levels.csv", index=False)
    pd.DataFrame(witness_rows).to_csv(out / "witness_index.csv", index=False)

    masks_by_s: Dict[float, List[np.ndarray]] = {
        s: [
            rq.independent_mask(n, s, args.seed + int(round(s * 1e9)) + 1000 * mi)
            for mi in range(args.mask_seeds)
        ]
        for s in selectivities
    }
    rq.estimator_table(levels_raw, masks_by_s).to_csv(out / "selectivity_estimator.csv", index=False)

    index.hnsw.efSearch = 64
    _, entry_ids = index.search(xq, 1)
    entry_ids = entry_ids[:, 0].astype(np.int32)

    max_pool = max(post_pools)
    index.hnsw.efSearch = max_pool
    tpost = time.perf_counter()
    _, post_candidates = index.search(xq, max_pool)
    post_ms_per_query = 1000.0 * (time.perf_counter() - tpost) / len(xq)

    rows: List[Dict[str, object]] = []
    materialization_rows: List[Dict[str, object]] = []
    structural_rows: List[Dict[str, object]] = []
    qids_by_mask = {
        mi: [qid for qid in range(len(xq)) if qid % args.mask_seeds == mi]
        for mi in range(args.mask_seeds)
    }

    def graph_degree(ip: np.ndarray, active_mask: np.ndarray) -> Tuple[float, float]:
        deg = np.diff(ip)[active_mask]
        if deg.size == 0:
            return 0.0, 1.0
        return float(deg.mean()), float(np.mean(deg == 0))

    for s in selectivities:
        level = choose_level_index(level_indices, s)
        native_level = min(
            (level_indices["L2"], level_indices["L3"]),
            key=lambda idx: abs(math.log((len(idx.portals) / n) / s)),
        )
        copies = level.copies[(args.bottom_b, 0)]
        copy_counts = [c for c in (1, 2, 4, 8) if c <= len(copies)]
        union_cache = {c: csr_union(copies[:c], len(level.portals)) for c in copy_counts}
        witness = witnesses[level.name]
        upper_copy = witness_upper_copy[level.name]
        log(
            f"s={s:g}: theory-matched={level.name} rate={len(level.portals)/n:.6f}; "
            f"native={native_level.name} rate={len(native_level.portals)/n:.6f}"
        )

        for mi, mask in enumerate(masks_by_s[s]):
            qualified, qoff, counts = rq.grouped_ids(level.cells, mask, len(level.portals))
            active = counts > 0
            valid_count = int(qualified.size)

            graph_variants: Dict[str, Tuple[np.ndarray, np.ndarray, float, int, float, float]] = {}
            for c in copy_counts:
                tmat = time.perf_counter()
                ip, ix, checks = filter_active_csr(union_cache[c][0], union_cache[c][1], active)
                mat_ms = 1000.0 * (time.perf_counter() - tmat)
                mean_deg, zero_deg = graph_degree(ip, active)
                graph_variants[f"active_c{c}"] = (ip, ix, mat_ms, checks, float(c), mean_deg)
                materialization_rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "level_name": level.name,
                        "method": f"active_c{c}",
                        "active_cells": int(active.sum()),
                        "edges": int(ix.size),
                        "mean_active_degree": mean_deg,
                        "zero_active_degree": zero_deg,
                        "materialization_ms": mat_ms,
                        "summary_checks": checks,
                        "copies_used_mean": float(c),
                    }
                )

            tmat = time.perf_counter()
            ad_ip, ad_ix, ad_checks, copies_used, ad_degree = adaptive_copy_graph(
                copies,
                active,
                args.degree_target,
                args.adaptive_max_copies,
            )
            ad_ms = 1000.0 * (time.perf_counter() - tmat)
            _, ad_zero = graph_degree(ad_ip, active)
            graph_variants["adaptive_deg"] = (
                ad_ip,
                ad_ix,
                ad_ms,
                ad_checks,
                copies_used,
                ad_degree,
            )
            materialization_rows.append(
                {
                    "selectivity": s,
                    "mask_seed": mi,
                    "level_name": level.name,
                    "method": "adaptive_deg",
                    "active_cells": int(active.sum()),
                    "edges": int(ad_ix.size),
                    "mean_active_degree": ad_degree,
                    "zero_active_degree": ad_zero,
                    "materialization_ms": ad_ms,
                    "summary_checks": ad_checks,
                    "copies_used_mean": copies_used,
                }
            )

            tmat = time.perf_counter()
            rt_ip, rt_ix, rt_checks, router_count = certified_router_csr(
                copies[0][0], copies[0][1], active, min_active_fanout=1
            )
            rt_ms = 1000.0 * (time.perf_counter() - tmat)
            rt_degree, rt_zero = graph_degree(rt_ip, active)
            graph_variants["router1hop"] = (rt_ip, rt_ix, rt_ms, rt_checks, 1.0, rt_degree)
            materialization_rows.append(
                {
                    "selectivity": s,
                    "mask_seed": mi,
                    "level_name": level.name,
                    "method": "router1hop",
                    "active_cells": int(active.sum()),
                    "router_cells": router_count,
                    "edges": int(rt_ix.size),
                    "mean_active_degree": rt_degree,
                    "zero_active_degree": rt_zero,
                    "materialization_ms": rt_ms,
                    "summary_checks": rt_checks,
                    "copies_used_mean": 1.0,
                }
            )

            # Native-tier active C1 baseline quantifies the virtual-tier gain.
            native_graph = None
            native_grouped = None
            native_mat_ms = 0.0
            native_checks = 0
            if native_level.name != level.name:
                nq, noff, ncounts = rq.grouped_ids(
                    native_level.cells, mask, len(native_level.portals)
                )
                nactive = ncounts > 0
                nt = time.perf_counter()
                nip, nix, native_checks = filter_active_csr(
                    native_level.copies[(args.bottom_b, 0)][0][0],
                    native_level.copies[(args.bottom_b, 0)][0][1],
                    nactive,
                )
                native_mat_ms = 1000.0 * (time.perf_counter() - nt)
                native_grouped = (nq, noff, ncounts, nactive)
                native_graph = (nip, nix)
                ndeg, nzero = graph_degree(nip, nactive)
                materialization_rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "level_name": native_level.name,
                        "method": "native_active_c1",
                        "active_cells": int(nactive.sum()),
                        "edges": int(nix.size),
                        "mean_active_degree": ndeg,
                        "zero_active_degree": nzero,
                        "materialization_ms": native_mat_ms,
                        "summary_checks": native_checks,
                        "copies_used_mean": 1.0,
                    }
                )

            counts_upper = np.bincount(
                witness.parent,
                weights=counts,
                minlength=len(witness.upper_portals),
            ).astype(np.int32)
            upper_active = counts_upper > 0
            tup = time.perf_counter()
            uip, uix, upper_checks = filter_active_csr(
                upper_copy[0], upper_copy[1], upper_active
            )
            upper_mat_ms = 1000.0 * (time.perf_counter() - tup)

            point_ids = np.flatnonzero(mask).astype(np.int32)
            block = adj[point_ids]
            safe = np.maximum(block, 0)
            point_deg = np.sum((block >= 0) & mask[safe], axis=1)
            for method, (ip, ix, _, _, used, mean_deg) in graph_variants.items():
                _, zero_deg = graph_degree(ip, active)
                structural_rows.append(
                    {
                        "selectivity": s,
                        "mask_seed": mi,
                        "level_name": level.name,
                        "method": method,
                        "valid_count": valid_count,
                        "active_cells": int(active.sum()),
                        "point_valid_degree": float(point_deg.mean()),
                        "point_zero_degree": float(np.mean(point_deg == 0)),
                        "active_degree": mean_deg,
                        "active_zero_degree": zero_deg,
                        "copies_used_mean": used,
                    }
                )

            for qid in qids_by_mask[mi]:
                q = xq[qid]
                truth = rq.topk_ids(xb, q, qualified, 10)
                tpre = time.perf_counter()
                _ = rq.topk_ids(xb, q, qualified, 10)
                rows.append(
                    {
                        "method": "prefilter_exact",
                        "selectivity": s,
                        "mask_seed": mi,
                        "query": qid,
                        "parameter": 0,
                        "level_name": level.name,
                        "recall": 1.0,
                        "distance_computations": valid_count,
                        "portal_computations": 0,
                        "payload_computations": valid_count,
                        "adjacency_reads": 0,
                        "visited_cells": 0,
                        "summary_checks_query": 0,
                        "child_reads_query": 0,
                        "latency_ms": 1000.0 * (time.perf_counter() - tpre),
                        "valid_count": valid_count,
                        "materialization_ms": 0.0,
                        "materialization_summary_checks": 0,
                        "copies_used_mean": 0.0,
                        "active_degree_mean": 0.0,
                        "postfilter_candidate_pool_lower_bound": 0,
                    }
                )
                for pool in post_pools:
                    cand = post_candidates[qid, :pool]
                    found = cand[mask[cand]][:10].astype(np.int32)
                    rows.append(
                        {
                            "method": "postfilter_hnsw",
                            "selectivity": s,
                            "mask_seed": mi,
                            "query": qid,
                            "parameter": pool,
                            "level_name": level.name,
                            "recall": rq.recall_at_10(found, truth),
                            "distance_computations": np.nan,
                            "portal_computations": np.nan,
                            "payload_computations": int(found.size),
                            "adjacency_reads": np.nan,
                            "visited_cells": np.nan,
                            "summary_checks_query": 0,
                            "child_reads_query": 0,
                            "latency_ms": post_ms_per_query,
                            "valid_count": valid_count,
                            "materialization_ms": 0.0,
                            "materialization_summary_checks": 0,
                            "copies_used_mean": 0.0,
                            "active_degree_mean": 0.0,
                            "postfilter_candidate_pool_lower_bound": pool,
                        }
                    )

                pred_seeds = rq.predicate_seed_cells(
                    mask,
                    levels_raw,
                    level.cells,
                    args.seed + 91_001 + qid,
                    limit=args.seed_count,
                )
                entry_cell = int(level.cells[entry_ids[qid]])
                active_seeds = pred_seeds
                if active[entry_cell]:
                    active_seeds = np.unique(
                        np.concatenate((active_seeds, np.asarray([entry_cell], dtype=np.int32)))
                    )
                router_seeds = np.unique(
                    np.concatenate((pred_seeds, np.asarray([entry_cell], dtype=np.int32)))
                )

                for ef in efs:
                    for method, (ip, ix, mat_ms, checks, used, mean_deg) in graph_variants.items():
                        seeds = router_seeds if method == "router1hop" else active_seeds
                        sr = search_cells(xb, q, level.portals, ip, ix, seeds, ef)
                        ans, payload_n, payload_ms = payload_answer(
                            xb, q, qualified, qoff, sr.selected_cells
                        )
                        append_result(
                            rows,
                            method,
                            s,
                            mi,
                            qid,
                            ef,
                            level.name,
                            sr,
                            ans,
                            truth,
                            payload_n,
                            payload_ms,
                            valid_count,
                            mat_ms,
                            checks,
                            copies_used_mean=used,
                            active_degree_mean=mean_deg,
                        )

                    base = graph_variants["active_c1"]
                    lazy = lazy_degree_refine_search(
                        xb,
                        q,
                        level.portals,
                        base[0],
                        base[1],
                        level.full_indptr,
                        level.full_indices,
                        counts,
                        active_seeds,
                        ef,
                        args.degree_target,
                        args.lazy_max_extra,
                        args.capacity_lambda,
                    )
                    ans, payload_n, payload_ms = payload_answer(
                        xb, q, qualified, qoff, lazy.selected_cells
                    )
                    append_result(
                        rows,
                        "lazy_qcap",
                        s,
                        mi,
                        qid,
                        ef,
                        level.name,
                        lazy,
                        ans,
                        truth,
                        payload_n,
                        payload_ms,
                        valid_count,
                        base[2],
                        base[3],
                        copies_used_mean=1.0,
                        active_degree_mean=base[5],
                    )

                    if native_graph is not None and native_grouped is not None:
                        nq, noff, ncounts, nactive = native_grouped
                        nseed = rq.predicate_seed_cells(
                            mask,
                            levels_raw,
                            native_level.cells,
                            args.seed + 193_001 + qid,
                            limit=args.seed_count,
                        )
                        nentry = int(native_level.cells[entry_ids[qid]])
                        if nactive[nentry]:
                            nseed = np.unique(
                                np.concatenate((nseed, np.asarray([nentry], dtype=np.int32)))
                            )
                        nsr = search_cells(
                            xb,
                            q,
                            native_level.portals,
                            native_graph[0],
                            native_graph[1],
                            nseed,
                            ef,
                        )
                        nans, npayload, npayload_ms = payload_answer(
                            xb, q, nq, noff, nsr.selected_cells
                        )
                        ndeg, _ = graph_degree(native_graph[0], nactive)
                        append_result(
                            rows,
                            "native_active_c1",
                            s,
                            mi,
                            qid,
                            ef,
                            native_level.name,
                            nsr,
                            nans,
                            truth,
                            npayload,
                            npayload_ms,
                            valid_count,
                            native_mat_ms,
                            native_checks,
                            copies_used_mean=1.0,
                            active_degree_mean=ndeg,
                        )

                    upper_seed = np.unique(witness.parent[active_seeds]).astype(np.int32)
                    upper_entry = int(witness.parent[entry_cell])
                    if upper_active[upper_entry]:
                        upper_seed = np.unique(
                            np.concatenate((upper_seed, np.asarray([upper_entry], dtype=np.int32)))
                        )
                    upper_ef = min(len(witness.upper_portals), max(8, min(args.upper_ef, ef // 4)))
                    upper_sr = search_cells(
                        xb,
                        q,
                        witness.upper_portals,
                        uip,
                        uix,
                        upper_seed,
                        upper_ef,
                    )
                    witness_seeds, wcomp, wchecks, wreads, wms = witness_descent_seeds(
                        xb,
                        q,
                        level,
                        witness,
                        (uip, uix),
                        upper_sr.selected_cells,
                        counts,
                        seed_budget=max(args.seed_count * 2, ef // 4),
                    )
                    target_seed = np.unique(
                        np.concatenate((active_seeds, witness_seeds))
                    ).astype(np.int32)
                    wsr = search_cells(
                        xb,
                        q,
                        level.portals,
                        base[0],
                        base[1],
                        target_seed,
                        ef,
                    )
                    wans, wpayload, wpayload_ms = payload_answer(
                        xb, q, qualified, qoff, wsr.selected_cells
                    )
                    append_result(
                        rows,
                        "witness_c1",
                        s,
                        mi,
                        qid,
                        ef,
                        level.name,
                        wsr,
                        wans,
                        truth,
                        wpayload,
                        wpayload_ms,
                        valid_count,
                        base[2] + upper_mat_ms,
                        base[3] + upper_checks,
                        extra_portal_computations=upper_sr.portal_computations + wcomp,
                        extra_adjacency_reads=upper_sr.adjacency_reads,
                        extra_summary_checks=wchecks,
                        extra_child_reads=wreads,
                        extra_latency_ms=upper_sr.latency_ms + wms,
                        copies_used_mean=1.0,
                        active_degree_mean=base[5],
                    )

    qdf = pd.DataFrame(rows)
    qdf.to_csv(out / "query_results.csv", index=False)
    mdf = pd.DataFrame(materialization_rows)
    mdf.to_csv(out / "materialization.csv", index=False)
    sdf = pd.DataFrame(structural_rows)
    sdf.to_csv(out / "structural.csv", index=False)

    summary = (
        qdf.groupby(["method", "selectivity", "parameter", "level_name"], as_index=False)
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
            copies_used_mean=("copies_used_mean", "mean"),
            active_degree_mean=("active_degree_mean", "mean"),
        )
        .sort_values(["selectivity", "method", "parameter"])
    )
    summary.to_csv(out / "summary.csv", index=False)

    theory_rows: List[Dict[str, object]] = []
    log_term = math.log(args.path_hops / args.path_delta)
    for s in selectivities:
        idx = choose_level_index(level_indices, s)
        B = n / len(idx.portals)
        tau = s * B
        rho = 1.0 - math.exp(-tau)
        copies_for_active = math.ceil(log_term / max(args.bottom_b * tau, 1e-12))
        theory_rows.append(
            {
                "selectivity": s,
                "level_name": idx.name,
                "cell_size_B": B,
                "occupancy_tau_sB": tau,
                "active_probability_approx": rho,
                "path_hops": args.path_hops,
                "path_delta": args.path_delta,
                "log_H_over_delta": log_term,
                "copies_bound_active_exit": copies_for_active,
                "configured_max_copies": args.max_copies,
                "degree_target": args.degree_target,
            }
        )
    pd.DataFrame(theory_rows).to_csv(out / "theory_guidance.csv", index=False)

    metadata = {
        "dataset": args.dataset,
        "n": n,
        "d": d,
        "queries": len(xq),
        "selectivities": selectivities,
        "efs": efs,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "bottom_b": args.bottom_b,
        "max_copies": args.max_copies,
        "adaptive_max_copies": args.adaptive_max_copies,
        "degree_target": args.degree_target,
        "lazy_max_extra": args.lazy_max_extra,
        "capacity_lambda": args.capacity_lambda,
        "virtual_rate_hi": args.virtual_rate_hi,
        "virtual_rate_lo": args.virtual_rate_lo,
        "build_seconds": build_seconds,
        "base_actual_edges": base_actual_edges,
        "hnsw_neighbor_slots": hnsw_neighbor_slots,
        "postfilter_ms_per_query": post_ms_per_query,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"complete: rows={len(qdf):,}, summary={len(summary):,}, out={out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/sift-128-euclidean.hdf5")
    p.add_argument("--out", default="results/rankq_v2")
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--ef-construction", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--mask-seeds", type=int, default=3)
    p.add_argument("--selectivities", default="0.001,0.002,0.003,0.005,0.007,0.01,0.02")
    p.add_argument("--efs", default="32,64,128,256")
    p.add_argument("--postfilter-pools", default="500,1000,2000,4000,8000,16000")
    p.add_argument("--bottom-b", type=int, default=8)
    p.add_argument("--max-copies", type=int, default=8)
    p.add_argument("--adaptive-max-copies", type=int, default=4)
    p.add_argument("--degree-target", type=int, default=8)
    p.add_argument("--lazy-max-extra", type=int, default=8)
    p.add_argument("--capacity-lambda", type=float, default=0.5)
    p.add_argument("--seed-count", type=int, default=4)
    p.add_argument("--upper-ef", type=int, default=32)
    p.add_argument("--virtual-rate-hi", type=float, default=0.008)
    p.add_argument("--virtual-rate-lo", type=float, default=0.004)
    p.add_argument("--path-hops", type=int, default=20)
    p.add_argument("--path-delta", type=float, default=0.01)
    run(p.parse_args())

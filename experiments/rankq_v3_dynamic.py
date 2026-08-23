from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import faiss
import h5py
import numpy as np
import pandas as pd

import sift1m_1pct_experiment as rq
import rankq_v2_progressive as v2


@dataclass
class DynamicSearchResult:
    selected_cells: np.ndarray
    portal_computations: int
    adjacency_reads: int
    visited_cells: int
    summary_checks: int
    copy_lists_read: int
    expanded_active_cells: int
    active_edges_exposed: int
    improving_edges: int
    geo_fallbacks: int
    router_fallbacks: int
    latency_ms: float


@dataclass
class NativeLevel:
    name: str
    portals: np.ndarray
    cells: np.ndarray
    full_indptr: np.ndarray
    full_indices: np.ndarray
    copies: List[Tuple[np.ndarray, np.ndarray]]
    geo_indptr: np.ndarray
    geo_indices: np.ndarray
    assignment_seconds: float
    contract_seconds: float
    geo_seconds: float
    cell_stats: Dict[str, float]


def log(msg: str) -> None:
    rq.log(msg)


def choose_native_level(levels: Mapping[str, NativeLevel], s: float, n: int) -> NativeLevel:
    return min(levels.values(), key=lambda idx: abs(math.log((len(idx.portals) / n) / s)))


def build_native_level(
    name: str,
    xb: np.ndarray,
    adj: np.ndarray,
    portals: np.ndarray,
    threads: int,
    b: int,
    max_copies: int,
    geo_core: int,
    seed: int,
) -> NativeLevel:
    cells, assign_s, stats = rq.assign_geometric_cells(xb, portals, threads)
    full_ip, full_ix, contract_s = rq.contract_graph(adj, cells, len(portals))
    copies: List[Tuple[np.ndarray, np.ndarray]] = []
    for ci in range(max_copies):
        ranks, _ = rq.edge_randoms(full_ip, full_ix, seed + 1_000_003 * ci)
        copies.append(rq.uniform_bottomk_graph(full_ip, full_ix, ranks, len(portals), b))
    geo_ip, geo_ix, geo_s = rq.geometric_core_graph(
        xb, portals, full_ip, full_ix, geo_core
    )
    return NativeLevel(
        name=name,
        portals=portals,
        cells=cells,
        full_indptr=full_ip,
        full_indices=full_ix,
        copies=copies,
        geo_indptr=geo_ip,
        geo_indices=geo_ix,
        assignment_seconds=assign_s,
        contract_seconds=contract_s,
        geo_seconds=geo_s,
        cell_stats=stats,
    )


def active_fanout(
    indptr: np.ndarray,
    indices: np.ndarray,
    active: np.ndarray,
) -> Tuple[np.ndarray, int]:
    n = len(active)
    fanout = np.zeros(n, dtype=np.int32)
    checks = 0
    for u in range(n):
        a, z = int(indptr[u]), int(indptr[u + 1])
        nb = indices[a:z]
        checks += int(nb.size)
        fanout[u] = int(np.count_nonzero(active[nb]))
    return fanout, checks


def expose_active_neighbors(
    copies: Sequence[Tuple[np.ndarray, np.ndarray]],
    active: np.ndarray,
    u: int,
    fixed_copies: int | None,
    degree_target: int,
    max_copies: int,
) -> Tuple[np.ndarray, int, int, int]:
    parts: List[np.ndarray] = []
    unique = np.empty(0, dtype=np.int32)
    reads = 0
    checks = 0
    used = 0
    cap = min(max_copies, len(copies))
    if fixed_copies is not None:
        cap = min(cap, fixed_copies)
    for ip, ix in copies[:cap]:
        a, z = int(ip[u]), int(ip[u + 1])
        nb = ix[a:z]
        reads += int(nb.size)
        checks += int(nb.size)
        good = nb[active[nb]]
        if good.size:
            parts.append(good)
            unique = np.unique(np.concatenate(parts)).astype(np.int32)
        used += 1
        if fixed_copies is None and unique.size >= degree_target:
            break
    return unique, reads, checks, used


def dynamic_search(
    xb: np.ndarray,
    q: np.ndarray,
    level: NativeLevel,
    counts: np.ndarray,
    seeds: np.ndarray,
    ef: int,
    fixed_copies: int | None,
    degree_target: int,
    max_copies: int,
    use_geo_on_stall: bool,
    use_router_on_stall: bool,
    router_fanout: np.ndarray | None,
    router_budget: int,
    progress_eps: float,
) -> DynamicSearchResult:
    t0 = time.perf_counter()
    active = counts > 0
    n = len(level.portals)
    ef = min(max(1, int(ef)), n)
    seeds = np.unique(seeds[(seeds >= 0) & (seeds < n)]).astype(np.int32)
    if not use_router_on_stall:
        seeds = seeds[active[seeds]]
    if seeds.size == 0:
        return DynamicSearchResult(
            np.empty(0, dtype=np.int32), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0
        )

    visited = np.zeros(n, dtype=np.uint8)
    distance_cache = np.full(n, np.nan, dtype=np.float32)
    ds0 = rq.l2_batch(xb[level.portals[seeds]], q)
    distance_cache[seeds] = ds0.astype(np.float32, copy=False)
    candidates: List[Tuple[float, int]] = []
    top: List[Tuple[float, int]] = []
    for dv, s0 in zip(ds0.tolist(), seeds.tolist()):
        visited[s0] = 1
        heapq.heappush(candidates, (float(dv), int(s0)))
        heapq.heappush(top, (-float(dv), int(s0)))
        if len(top) > ef:
            heapq.heappop(top)

    portal_comps = int(seeds.size)
    adjacency_reads = 0
    summary_checks = 0
    copy_lists_read = 0
    expanded_active = 0
    active_edges_exposed = 0
    improving_edges = 0
    geo_fallbacks = 0
    router_fallbacks = 0

    def ensure_dist(ids: np.ndarray) -> np.ndarray:
        nonlocal portal_comps
        if ids.size == 0:
            return np.empty(0, dtype=np.float32)
        out = distance_cache[ids].copy()
        miss = np.isnan(out)
        if np.any(miss):
            vals = rq.l2_batch(xb[level.portals[ids[miss]]], q)
            portal_comps += int(np.count_nonzero(miss))
            out[miss] = vals
            distance_cache[ids[miss]] = vals
        return out

    while candidates:
        du, u = heapq.heappop(candidates)
        if len(top) >= ef and du > -top[0][0]:
            break

        chosen = np.empty(0, dtype=np.int32)
        if active[u]:
            expanded_active += 1
            chosen, rd, ck, used = expose_active_neighbors(
                level.copies,
                active,
                u,
                fixed_copies=fixed_copies,
                degree_target=degree_target,
                max_copies=max_copies,
            )
            adjacency_reads += rd
            summary_checks += ck
            copy_lists_read += used
            active_edges_exposed += int(chosen.size)
            chosen_ds = ensure_dist(chosen)
            progress_mask = chosen_ds < float(du) * (1.0 - progress_eps)
            improving_edges += int(np.count_nonzero(progress_mask))

            if use_geo_on_stall and not np.any(progress_mask):
                ga, gz = int(level.geo_indptr[u]), int(level.geo_indptr[u + 1])
                geo = level.geo_indices[ga:gz]
                adjacency_reads += int(geo.size)
                summary_checks += int(geo.size)
                geo = geo[active[geo]]
                if geo.size:
                    geo_fallbacks += 1
                    chosen = np.unique(np.concatenate((chosen, geo))).astype(np.int32)
                    active_edges_exposed += int(geo.size)
                    chosen_ds = ensure_dist(chosen)
                    progress_mask = chosen_ds < float(du) * (1.0 - progress_eps)
                    improving_edges += int(np.count_nonzero(progress_mask))

            if use_router_on_stall and not np.any(progress_mask) and router_fanout is not None:
                a, z = int(level.copies[0][0][u]), int(level.copies[0][0][u + 1])
                nb = level.copies[0][1][a:z]
                adjacency_reads += int(nb.size)
                summary_checks += int(nb.size)
                routers = nb[(~active[nb]) & (router_fanout[nb] > 0)]
                if routers.size:
                    rds = ensure_dist(routers)
                    keep = min(router_budget, routers.size)
                    if routers.size > keep:
                        pos = np.argpartition(rds, keep - 1)[:keep]
                        routers = routers[pos]
                    chosen = np.unique(np.concatenate((chosen, routers))).astype(np.int32)
                    router_fallbacks += 1
        elif use_router_on_stall:
            a, z = int(level.copies[0][0][u]), int(level.copies[0][0][u + 1])
            nb = level.copies[0][1][a:z]
            adjacency_reads += int(nb.size)
            summary_checks += int(nb.size)
            chosen = nb[active[nb]]
            active_edges_exposed += int(chosen.size)

        if chosen.size == 0:
            continue
        unseen = chosen[visited[chosen] == 0]
        if unseen.size == 0:
            continue
        visited[unseen] = 1
        ds = ensure_dist(unseen)
        for dv, v0 in zip(ds.tolist(), unseen.tolist()):
            v = int(v0)
            if len(top) < ef or dv < -top[0][0]:
                heapq.heappush(candidates, (float(dv), v))
                heapq.heappush(top, (-float(dv), v))
                if len(top) > ef:
                    heapq.heappop(top)

    selected = np.asarray(
        [u for _, u in sorted([(-negd, u) for negd, u in top]) if active[u]],
        dtype=np.int32,
    )
    return DynamicSearchResult(
        selected_cells=selected,
        portal_computations=portal_comps,
        adjacency_reads=adjacency_reads,
        visited_cells=int(np.count_nonzero(visited)),
        summary_checks=summary_checks,
        copy_lists_read=copy_lists_read,
        expanded_active_cells=expanded_active,
        active_edges_exposed=active_edges_exposed,
        improving_edges=improving_edges,
        geo_fallbacks=geo_fallbacks,
        router_fallbacks=router_fallbacks,
        latency_ms=1000.0 * (time.perf_counter() - t0),
    )


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
    result: DynamicSearchResult,
    answer: np.ndarray,
    truth: np.ndarray,
    payload_n: int,
    payload_ms: float,
    valid_count: int,
    theory_copies: int,
) -> None:
    avg_copies = (
        result.copy_lists_read / result.expanded_active_cells
        if result.expanded_active_cells > 0
        else 0.0
    )
    progress_fraction = (
        result.improving_edges / result.active_edges_exposed
        if result.active_edges_exposed > 0
        else 0.0
    )
    rows.append(
        {
            "method": method,
            "selectivity": selectivity,
            "mask_seed": mask_seed,
            "query": qid,
            "parameter": parameter,
            "level_name": level_name,
            "recall": rq.recall_at_10(answer, truth),
            "distance_computations": result.portal_computations + payload_n,
            "portal_computations": result.portal_computations,
            "payload_computations": payload_n,
            "adjacency_reads": result.adjacency_reads,
            "visited_cells": result.visited_cells,
            "summary_checks_query": result.summary_checks,
            "copy_lists_read": result.copy_lists_read,
            "expanded_active_cells": result.expanded_active_cells,
            "avg_copies_per_expansion": avg_copies,
            "active_edges_exposed": result.active_edges_exposed,
            "improving_edges": result.improving_edges,
            "progress_fraction": progress_fraction,
            "geo_fallbacks": result.geo_fallbacks,
            "router_fallbacks": result.router_fallbacks,
            "latency_ms": result.latency_ms + payload_ms,
            "valid_count": valid_count,
            "theory_copies": theory_copies,
        }
    )


def theory_copy_bound(
    s: float,
    cell_size: float,
    b: int,
    path_hops: int,
    delta: float,
) -> int:
    tau = s * cell_size
    if tau <= 0:
        return 1
    return max(1, int(math.ceil(math.log(path_hops / delta) / (b * tau))))


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
    log(f"loaded SIFT1M: xb={xb.shape}, queries={len(xq)}, selectivities={selectivities}")

    index, build_seconds = rq.build_hnsw(xb, args.M, args.ef_construction, args.threads)
    levels_raw, adj, base_actual_edges, hnsw_neighbor_slots = rq.extract_layer0(index, n)

    native_levels: Dict[str, NativeLevel] = {}
    index_rows: List[Dict[str, object]] = []
    for li, physical_level in enumerate((2, 3)):
        name = f"L{physical_level}"
        portals = rq.portal_ids(levels_raw, physical_level)
        log(f"build {name}: portals={len(portals):,}, rate={len(portals)/n:.6f}")
        level = build_native_level(
            name,
            xb,
            adj,
            portals,
            args.threads,
            args.bottom_b,
            args.max_copies,
            args.geo_core,
            args.seed + 100_000 * li,
        )
        native_levels[name] = level
        index_rows.append(
            {
                "level_name": name,
                "n_cells": len(level.portals),
                "sampling_rate": len(level.portals) / n,
                "mean_cell_size": n / len(level.portals),
                "assignment_seconds": level.assignment_seconds,
                "contract_seconds": level.contract_seconds,
                "geo_seconds": level.geo_seconds,
                "full_edges": int(level.full_indices.size),
                "copy1_edges": int(level.copies[0][1].size),
                "copy2_edge_sum": int(sum(g[1].size for g in level.copies[:2])),
                "copy4_edge_sum": int(sum(g[1].size for g in level.copies[:4])),
                **{f"cell_{k}": v for k, v in level.cell_stats.items()},
            }
        )
    pd.DataFrame(index_rows).to_csv(out / "index_levels.csv", index=False)

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
    structural_rows: List[Dict[str, object]] = []
    qids_by_mask = {
        mi: [qid for qid in range(len(xq)) if qid % args.mask_seeds == mi]
        for mi in range(args.mask_seeds)
    }

    for s in selectivities:
        level = choose_native_level(native_levels, s, n)
        B = n / len(level.portals)
        c_theory = min(
            args.max_copies,
            theory_copy_bound(s, B, args.bottom_b, args.path_hops, args.path_delta),
        )
        tau = s * B
        rho = 1.0 - math.exp(-tau)
        log(
            f"s={s:g}: level={level.name}, B={B:.2f}, tau={tau:.3f}, "
            f"rho={rho:.3f}, theory copies={c_theory}"
        )

        for mi, mask in enumerate(masks_by_s[s]):
            qualified, qoff, counts = rq.grouped_ids(level.cells, mask, len(level.portals))
            active = counts > 0
            valid_count = int(qualified.size)
            fanout, fanout_checks = active_fanout(
                level.copies[0][0], level.copies[0][1], active
            )

            point_ids = np.flatnonzero(mask).astype(np.int32)
            block = adj[point_ids]
            safe = np.maximum(block, 0)
            point_deg = np.sum((block >= 0) & mask[safe], axis=1)
            structural_rows.append(
                {
                    "selectivity": s,
                    "mask_seed": mi,
                    "level_name": level.name,
                    "valid_count": valid_count,
                    "active_cells": int(active.sum()),
                    "point_valid_degree": float(point_deg.mean()),
                    "point_zero_degree": float(np.mean(point_deg == 0)),
                    "copy1_active_degree": float(fanout[active].mean()) if np.any(active) else 0.0,
                    "copy1_zero_active_degree": float(np.mean(fanout[active] == 0)) if np.any(active) else 1.0,
                    "fanout_summary_checks": fanout_checks,
                    "theory_copies": c_theory,
                    "tau_sB": tau,
                    "active_probability_approx": rho,
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
                        "copy_lists_read": 0,
                        "expanded_active_cells": 0,
                        "avg_copies_per_expansion": 0.0,
                        "active_edges_exposed": 0,
                        "improving_edges": 0,
                        "progress_fraction": 0.0,
                        "geo_fallbacks": 0,
                        "router_fallbacks": 0,
                        "latency_ms": 1000.0 * (time.perf_counter() - tpre),
                        "valid_count": valid_count,
                        "theory_copies": c_theory,
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
                            "copy_lists_read": 0,
                            "expanded_active_cells": 0,
                            "avg_copies_per_expansion": 0.0,
                            "active_edges_exposed": 0,
                            "improving_edges": 0,
                            "progress_fraction": 0.0,
                            "geo_fallbacks": 0,
                            "router_fallbacks": 0,
                            "latency_ms": post_ms_per_query,
                            "valid_count": valid_count,
                            "theory_copies": c_theory,
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

                method_specs = [
                    ("fixed_c1", 1, 0, False, False, active_seeds),
                    ("fixed_c2", min(2, args.max_copies), 0, False, False, active_seeds),
                    ("theory_c", c_theory, 0, False, False, active_seeds),
                    ("degree4", None, 4, False, False, active_seeds),
                    ("degree8", None, 8, False, False, active_seeds),
                    ("progress4", None, 4, True, False, active_seeds),
                    ("progress8", None, 8, True, False, active_seeds),
                    ("progress4_router", None, 4, True, True, router_seeds),
                ]

                for ef in efs:
                    for method, fixed_c, degree_target, use_geo, use_router, seeds in method_specs:
                        target = degree_target if fixed_c is None else 0
                        result = dynamic_search(
                            xb,
                            q,
                            level,
                            counts,
                            seeds,
                            ef,
                            fixed_copies=fixed_c,
                            degree_target=target,
                            max_copies=args.controller_max_copies,
                            use_geo_on_stall=use_geo,
                            use_router_on_stall=use_router,
                            router_fanout=fanout if use_router else None,
                            router_budget=args.router_budget,
                            progress_eps=args.progress_eps,
                        )
                        answer, payload_n, payload_ms = payload_answer(
                            xb, q, qualified, qoff, result.selected_cells
                        )
                        append_result(
                            rows,
                            method,
                            s,
                            mi,
                            qid,
                            ef,
                            level.name,
                            result,
                            answer,
                            truth,
                            payload_n,
                            payload_ms,
                            valid_count,
                            c_theory,
                        )

    qdf = pd.DataFrame(rows)
    qdf.to_csv(out / "query_results.csv", index=False)
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(out / "structural.csv", index=False)

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
            copy_lists_read=("copy_lists_read", "mean"),
            expanded_active_cells=("expanded_active_cells", "mean"),
            avg_copies_per_expansion=("avg_copies_per_expansion", "mean"),
            active_edges_exposed=("active_edges_exposed", "mean"),
            improving_edges=("improving_edges", "mean"),
            progress_fraction=("progress_fraction", "mean"),
            geo_fallbacks=("geo_fallbacks", "mean"),
            router_fallbacks=("router_fallbacks", "mean"),
            latency_ms=("latency_ms", "mean"),
            valid_count=("valid_count", "mean"),
            theory_copies=("theory_copies", "mean"),
        )
        .sort_values(["selectivity", "method", "parameter"])
    )
    summary.to_csv(out / "summary.csv", index=False)

    theory_rows = []
    for s in selectivities:
        level = choose_native_level(native_levels, s, n)
        B = n / len(level.portals)
        tau = s * B
        rho = 1.0 - math.exp(-tau)
        c = theory_copy_bound(s, B, args.bottom_b, args.path_hops, args.path_delta)
        theory_rows.append(
            {
                "selectivity": s,
                "level_name": level.name,
                "cell_size_B": B,
                "tau_sB": tau,
                "active_probability_approx": rho,
                "path_hops": args.path_hops,
                "path_delta": args.path_delta,
                "log_H_over_delta": math.log(args.path_hops / args.path_delta),
                "theory_copy_bound": c,
                "degree_target4": 4,
                "degree_target8": 8,
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
        "controller_max_copies": args.controller_max_copies,
        "geo_core": args.geo_core,
        "progress_eps": args.progress_eps,
        "router_budget": args.router_budget,
        "build_seconds": build_seconds,
        "base_actual_edges": base_actual_edges,
        "hnsw_neighbor_slots": hnsw_neighbor_slots,
        "postfilter_ms_per_query": post_ms_per_query,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"complete: query rows={len(qdf):,}, summary rows={len(summary):,}, out={out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/sift-128-euclidean.hdf5")
    p.add_argument("--out", default="results/rankq_v3")
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
    p.add_argument("--max-copies", type=int, default=4)
    p.add_argument("--controller-max-copies", type=int, default=4)
    p.add_argument("--geo-core", type=int, default=8)
    p.add_argument("--progress-eps", type=float, default=0.01)
    p.add_argument("--router-budget", type=int, default=2)
    p.add_argument("--seed-count", type=int, default=4)
    p.add_argument("--path-hops", type=int, default=20)
    p.add_argument("--path-delta", type=float, default=0.01)
    run(p.parse_args())

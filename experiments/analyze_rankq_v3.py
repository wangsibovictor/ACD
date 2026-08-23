from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "(no rows)"
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in x.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def first_reaching(g: pd.DataFrame, threshold: float) -> pd.Series | None:
    good = g[g["recall"] >= threshold].copy()
    if good.empty:
        return None
    if good["distance_computations"].notna().any():
        return good.sort_values(["distance_computations", "adjacency_reads", "parameter"]).iloc[0]
    return good.sort_values("parameter").iloc[0]


def main() -> None:
    root = Path("experiment_outputs/rankq_v3")
    summary = pd.read_csv(root / "summary.csv")
    structural = pd.read_csv(root / "structural.csv")
    theory = pd.read_csv(root / "theory_guidance.csv")
    levels = pd.read_csv(root / "index_levels.csv")
    metadata = json.loads((root / "metadata.json").read_text())

    graph_methods = [
        x for x in summary["method"].unique()
        if x not in {"prefilter_exact", "postfilter_hnsw"}
    ]
    selectivities = sorted(summary["selectivity"].unique())

    best_rows = []
    for threshold in (0.95, 0.99):
        for s in selectivities:
            sg = summary[summary["selectivity"] == s]
            pre = sg[sg["method"] == "prefilter_exact"].iloc[0]
            post = first_reaching(sg[sg["method"] == "postfilter_hnsw"], threshold)
            cand = []
            for method in graph_methods:
                r = first_reaching(sg[sg["method"] == method], threshold)
                if r is not None:
                    cand.append(r)
            best = min(cand, key=lambda r: (r["distance_computations"], r["adjacency_reads"])) if cand else None
            best_rows.append({
                "threshold": threshold,
                "selectivity": s,
                "best_graph_method": "none" if best is None else best["method"],
                "level_name": "" if best is None else best["level_name"],
                "parameter": np.nan if best is None else best["parameter"],
                "graph_recall": np.nan if best is None else best["recall"],
                "graph_distances": np.nan if best is None else best["distance_computations"],
                "graph_edges": np.nan if best is None else best["adjacency_reads"],
                "avg_copies": np.nan if best is None else best["avg_copies_per_expansion"],
                "progress_fraction": np.nan if best is None else best["progress_fraction"],
                "geo_fallbacks": np.nan if best is None else best["geo_fallbacks"],
                "router_fallbacks": np.nan if best is None else best["router_fallbacks"],
                "exact_scan_distances": pre["distance_computations"],
                "postfilter_min_pool": np.nan if post is None else post["parameter"],
                "postfilter_recall": np.nan if post is None else post["recall"],
                "distance_reduction_vs_scan": np.nan if best is None else 1.0 - best["distance_computations"] / pre["distance_computations"],
                "distance_reduction_vs_post_pool_lb": np.nan if best is None or post is None else 1.0 - best["distance_computations"] / post["parameter"],
            })
    best = pd.DataFrame(best_rows)
    best.to_csv(root / "analysis_best_thresholds.csv", index=False)

    fixed_methods = [
        "fixed_c1", "fixed_c2", "theory_c", "degree4", "degree8",
        "progress4", "progress8", "progress4_router",
    ]
    fixed = summary[
        (summary["parameter"] == 128) & summary["method"].isin(fixed_methods)
    ][[
        "selectivity", "method", "level_name", "recall", "distance_computations",
        "adjacency_reads", "avg_copies_per_expansion", "progress_fraction",
        "geo_fallbacks", "router_fallbacks", "summary_checks_query", "latency_ms",
    ]].sort_values(["selectivity", "method"])
    fixed.to_csv(root / "analysis_fixed_ef128.csv", index=False)

    def pair(a: str, b: str, ef: int = 128) -> pd.DataFrame:
        aa = summary[(summary["method"] == a) & (summary["parameter"] == ef)].copy()
        bb = summary[(summary["method"] == b) & (summary["parameter"] == ef)].copy()
        m = aa.merge(bb, on="selectivity", suffixes=("_a", "_b"))
        if m.empty:
            return m
        return pd.DataFrame({
            "selectivity": m["selectivity"],
            "A": a,
            "B": b,
            "recall_A": m["recall_a"],
            "recall_B": m["recall_b"],
            "recall_delta": m["recall_a"] - m["recall_b"],
            "dist_A": m["distance_computations_a"],
            "dist_B": m["distance_computations_b"],
            "distance_ratio": m["distance_computations_a"] / m["distance_computations_b"],
            "edge_ratio": m["adjacency_reads_a"] / m["adjacency_reads_b"],
            "avg_copies_A": m["avg_copies_per_expansion_a"],
            "progress_A": m["progress_fraction_a"],
            "geo_fallbacks_A": m["geo_fallbacks_a"],
            "router_fallbacks_A": m["router_fallbacks_a"],
        })

    comparisons = []
    for a in ["fixed_c2", "theory_c", "degree4", "degree8", "progress4", "progress8", "progress4_router"]:
        x = pair(a, "fixed_c1")
        if not x.empty:
            comparisons.append(x)
    comp = pd.concat(comparisons, ignore_index=True)
    comp.to_csv(root / "analysis_pairwise_ef128.csv", index=False)

    # Theory/observation alignment.
    theory_obs = theory.merge(
        fixed[fixed["method"].isin(["fixed_c1", "fixed_c2", "theory_c", "degree4", "progress4"])],
        on="selectivity",
        suffixes=("_theory", "_observed"),
    )
    theory_obs.to_csv(root / "analysis_theory_observed.csv", index=False)

    b95 = best[best["threshold"] == 0.95]
    b99 = best[best["threshold"] == 0.99]
    lines = [
        "# RankQ v3 dynamic degree/progress controller: computed analysis",
        "",
        "## Best graph at Recall >= 0.95",
        "",
        md_table(b95, 4),
        "",
        "## Best graph at Recall >= 0.99",
        "",
        md_table(b99, 4),
        "",
        "## Fixed ef=128",
        "",
        md_table(fixed, 4),
        "",
        "## Pairwise versus fixed C1 at ef=128",
        "",
        md_table(comp, 4),
        "",
        "## Theory guidance",
        "",
        md_table(theory, 4),
        "",
        "## Structural measurements",
        "",
        md_table(structural, 4),
        "",
        "## Index levels",
        "",
        md_table(levels, 4),
        "",
    ]
    (root / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "ANALYSIS.md")


if __name__ == "__main__":
    main()

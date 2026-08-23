from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "(no rows)"
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{v:.{float_digits}f}")
    cols = list(x.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in x.itertuples(index=False, name=None):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def first_reaching(g: pd.DataFrame, threshold: float, cost_col: str = "distance_computations") -> pd.Series | None:
    good = g[g["recall"] >= threshold].copy()
    if good.empty:
        return None
    if good[cost_col].notna().any():
        return good.sort_values([cost_col, "parameter"]).iloc[0]
    return good.sort_values("parameter").iloc[0]


def main() -> None:
    root = Path("experiment_outputs/rankq_v2")
    summary = pd.read_csv(root / "summary.csv")
    material = pd.read_csv(root / "materialization.csv")
    levels = pd.read_csv(root / "index_levels.csv")
    witness = pd.read_csv(root / "witness_index.csv")
    theory = pd.read_csv(root / "theory_guidance.csv")
    metadata = json.loads((root / "metadata.json").read_text())

    graph_methods = [
        m for m in summary["method"].unique()
        if m not in {"prefilter_exact", "postfilter_hnsw"}
    ]
    selectivities = sorted(summary["selectivity"].unique())

    best_rows = []
    for threshold in (0.95, 0.99):
        for s in selectivities:
            sg = summary[summary["selectivity"] == s]
            pre = sg[sg["method"] == "prefilter_exact"].iloc[0]
            post = first_reaching(sg[sg["method"] == "postfilter_hnsw"], threshold)
            candidates = []
            for method in graph_methods:
                r = first_reaching(sg[sg["method"] == method], threshold)
                if r is not None:
                    candidates.append(r)
            best = min(candidates, key=lambda r: (r["distance_computations"], r["adjacency_reads"])) if candidates else None
            best_rows.append({
                "threshold": threshold,
                "selectivity": s,
                "best_graph_method": "none" if best is None else best["method"],
                "level_name": "" if best is None else best["level_name"],
                "parameter": np.nan if best is None else best["parameter"],
                "graph_recall": np.nan if best is None else best["recall"],
                "graph_distances": np.nan if best is None else best["distance_computations"],
                "graph_edges": np.nan if best is None else best["adjacency_reads"],
                "graph_materialization_ms": np.nan if best is None else best["materialization_ms"],
                "exact_scan_distances": pre["distance_computations"],
                "postfilter_min_pool": np.nan if post is None else post["parameter"],
                "postfilter_recall": np.nan if post is None else post["recall"],
                "distance_reduction_vs_scan": np.nan if best is None else 1.0 - best["distance_computations"] / pre["distance_computations"],
                "distance_reduction_vs_post_pool_lb": np.nan if best is None or post is None else 1.0 - best["distance_computations"] / post["parameter"],
            })
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(root / "analysis_best_thresholds.csv", index=False)

    fixed = summary[
        (summary["parameter"] == 128)
        & summary["method"].isin([
            "active_c1", "active_c2", "active_c4", "active_c8",
            "adaptive_deg", "lazy_qcap", "router1hop", "witness_c1",
            "native_active_c1",
        ])
    ][[
        "selectivity", "method", "level_name", "recall", "distance_computations",
        "adjacency_reads", "materialization_ms", "copies_used_mean", "active_degree_mean",
    ]].sort_values(["selectivity", "method"])
    fixed.to_csv(root / "analysis_fixed_ef128.csv", index=False)

    def pairwise(method_a: str, method_b: str, parameter: int = 128) -> pd.DataFrame:
        a = summary[(summary["method"] == method_a) & (summary["parameter"] == parameter)].copy()
        b = summary[(summary["method"] == method_b) & (summary["parameter"] == parameter)].copy()
        m = a.merge(b, on="selectivity", suffixes=("_a", "_b"))
        return pd.DataFrame({
            "selectivity": m["selectivity"],
            "A": method_a,
            "B": method_b,
            "recall_A": m["recall_a"],
            "recall_B": m["recall_b"],
            "recall_delta_A_minus_B": m["recall_a"] - m["recall_b"],
            "dist_A": m["distance_computations_a"],
            "dist_B": m["distance_computations_b"],
            "distance_ratio_A_over_B": m["distance_computations_a"] / m["distance_computations_b"],
            "edge_ratio_A_over_B": m["adjacency_reads_a"] / m["adjacency_reads_b"],
            "mat_ratio_A_over_B": m["materialization_ms_a"] / m["materialization_ms_b"],
        })

    comparisons = []
    for a, b in [
        ("active_c2", "active_c1"),
        ("active_c4", "active_c1"),
        ("active_c8", "active_c1"),
        ("adaptive_deg", "active_c1"),
        ("lazy_qcap", "active_c1"),
        ("router1hop", "active_c1"),
        ("witness_c1", "active_c1"),
        ("active_c1", "native_active_c1"),
    ]:
        c = pairwise(a, b)
        if not c.empty:
            comparisons.append(c)
    comp_df = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    comp_df.to_csv(root / "analysis_pairwise_ef128.csv", index=False)

    mat_avg = material.groupby(["selectivity", "method", "level_name"], as_index=False).agg(
        materialization_ms=("materialization_ms", "mean"),
        summary_checks=("summary_checks", "mean"),
        edges=("edges", "mean"),
        mean_active_degree=("mean_active_degree", "mean"),
        zero_active_degree=("zero_active_degree", "mean"),
        copies_used_mean=("copies_used_mean", "mean"),
    )
    mat_avg.to_csv(root / "analysis_materialization.csv", index=False)

    # Compact conclusions based on exact computed values.
    b95 = best_df[best_df["threshold"] == 0.95].copy()
    c2 = comp_df[(comp_df["A"] == "active_c2") & (comp_df["B"] == "active_c1")]
    c4 = comp_df[(comp_df["A"] == "active_c4") & (comp_df["B"] == "active_c1")]
    c8 = comp_df[(comp_df["A"] == "active_c8") & (comp_df["B"] == "active_c1")]
    cad = comp_df[(comp_df["A"] == "adaptive_deg") & (comp_df["B"] == "active_c1")]
    clazy = comp_df[(comp_df["A"] == "lazy_qcap") & (comp_df["B"] == "active_c1")]
    cw = comp_df[(comp_df["A"] == "witness_c1") & (comp_df["B"] == "active_c1")]
    cv = comp_df[(comp_df["A"] == "active_c1") & (comp_df["B"] == "native_active_c1")]

    lines = [
        "# RankQ v2 theory-guided optimization: computed analysis",
        "",
        "## Best tested graph configuration at Recall >= 0.95",
        "",
        md_table(b95[[
            "selectivity", "best_graph_method", "level_name", "parameter", "graph_recall",
            "graph_distances", "graph_edges", "exact_scan_distances", "postfilter_min_pool",
            "distance_reduction_vs_scan", "distance_reduction_vs_post_pool_lb",
        ]], 4),
        "",
        "## Fixed ef=128: copy scaling and adaptive methods",
        "",
        md_table(fixed, 4),
        "",
        "## Theory guidance",
        "",
        md_table(theory, 4),
        "",
        "## Index levels",
        "",
        md_table(levels, 4),
        "",
        "## Witness statistics",
        "",
        md_table(witness, 4),
        "",
        "## Aggregate pairwise observations (ef=128)",
        "",
    ]

    for title, df in [
        ("Active C2 versus C1", c2),
        ("Active C4 versus C1", c4),
        ("Active C8 versus C1", c8),
        ("Adaptive degree versus C1", cad),
        ("Lazy query-capacity refinement versus C1", clazy),
        ("Witness descent versus C1", cw),
        ("Virtual-tier C1 versus native-tier C1", cv),
    ]:
        lines += [f"### {title}", "", md_table(df, 4), ""]

    (root / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {root / 'ANALYSIS.md'}")


if __name__ == "__main__":
    main()

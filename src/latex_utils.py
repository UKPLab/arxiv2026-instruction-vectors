import numpy as np
import os

def comparisons_to_latex(
    all_comparisons,
    models,
    task,
    num_heads,
    num_layers,
    float_fmt=".2f",
    save_path=None,
):
    """
    Render compare_heatmaps results for a single task across multiple models as a
    LaTeX booktabs table. One row per model.

    Columns: spearman_overall, jaccard, spearman_per_layer, spearman_per_head,
             diff_heatmap (as a row-per-layer block of values).

    Requires LaTeX packages: booktabs, makecell, array.
    The diff_heatmap column is wide; wrap the table in a sidewaystable (rotating
    package) or \resizebox{\textwidth}{!}{...} for best results.

    Args:
        all_comparisons: {model_name: output of compare_heatmaps()}
        models:          ordered list of model names (determines row order)
        task:            task name used in the caption and label
        num_heads:       number of attention heads
        num_layers:      number of layers
        float_fmt:       Python format spec for float values (default ".2f")
        save_path:       if given, write the .tex snippet to this path

    Returns:
        str: LaTeX table source
    """
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return format(float(v), float_fmt)

    def fmt_list(lst):
        # Plain text — placed directly in a p{} column so it wraps at commas
        return r"{\scriptsize [" + ", ".join(fmt(v) for v in lst) + r"]}"

    def fmt_matrix(mat):
        # Nested tabular: & between columns, \\ between rows — values align
        row_strs = [" & ".join(fmt(v) for v in row) for row in mat]
        return (
            r"{\tiny\begin{tabular}[t]{@{}*{"
            + str(num_heads)
            + r"}{r}@{}}"
            + r" \\ ".join(row_strs)
            + r"\end{tabular}}"
        )

    col_spec = r"l c c p{5.5cm} p{5.5cm} p{9cm}"
    header = " & ".join([
        r"\textbf{Model}",
        r"$r_{\text{overall}}$",
        r"Jaccard",
        r"$r_{\text{per-layer}}$ {\scriptsize (layers 0--" + str(num_layers - 1) + ")}",
        r"$r_{\text{per-head}}$ {\scriptsize (heads 0--" + str(num_heads - 1) + ")}",
        r"Diff heatmap ($k_2 - k_1$, rows = layers)",
    ]) + r" \\"

    rows = []
    for model in models:
        cmp = all_comparisons.get(model)
        if cmp is None:
            continue
        cells = [
            model.replace("_", r"\_"),
            fmt(cmp["spearman_overall"]),
            fmt(cmp["jaccard"]),
            fmt_list(cmp["spearman_per_layer"]),
            fmt_list(cmp["spearman_per_head"]),
            fmt_matrix(cmp["diff_heatmap"]),
        ]
        rows.append(" & ".join(cells) + r" \\")

    safe_task = task.replace("_", "-")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Structural comparison (k=1 vs k=2) across models --- task: " + safe_task + "}",
        r"\label{tab:structural-comparison-" + safe_task + "}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")

    # ── Per-layer and per-head individual tables ───────────────────────────────
    def _index_table(metric_key, index_label, caption_str, label_str):
        n_cols = len(next(iter(all_comparisons.values()))[metric_key])
        index_rows = []
        for model in models:
            cmp = all_comparisons.get(model)
            if cmp is None:
                continue
            vals = " & ".join(fmt(v) for v in cmp[metric_key])
            index_rows.append(model.replace("_", r"\_") + " & " + vals + r" \\")
        return "\n".join([
            r"\begin{table}[h]",
            r"\centering",
            r"\begin{tabular}{l *{" + str(n_cols) + r"}{r}}",
            r"\toprule",
            r"\textbf{Model} & \multicolumn{" + str(n_cols) + r"}{c}{" + index_label + r"} \\",
            r" & " + " & ".join(str(i) for i in range(n_cols)) + r" \\",
            r"\midrule",
            *index_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{" + caption_str + " --- task: " + safe_task + "}",
            r"\label{" + label_str + "-" + safe_task + "}",
            r"\end{table}",
        ])

    latex_per_layer = _index_table(
        "spearman_per_layer",
        r"Spearman $r$ per layer (k=1 vs k=2)",
        r"Per-layer Spearman $r$ (k=1 vs k=2)",
        r"tab:per-layer",
    )
    latex_per_head = _index_table(
        "spearman_per_head",
        r"Spearman $r$ per head (k=1 vs k=2)",
        r"Per-head Spearman $r$ (k=1 vs k=2)",
        r"tab:per-head",
    )

    if save_path:
        base = save_path.replace("_comparison.tex", "")
        for suffix, content in [("_per_layer.tex", latex_per_layer), ("_per_head.tex", latex_per_head)]:
            path = base + suffix
            with open(path, "w") as f:
                f.write(content)
            print(f"LaTeX table saved to: {path}")

    return latex
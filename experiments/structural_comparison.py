import os
from collections import defaultdict

import pandas as pd
import torch
import numpy as np
from scipy.stats import beta as beta_dist, spearmanr, chi2 as chi2_dist
from tqdm import tqdm

from src import PROJECT_ROOT
from src.graph_utils import (
    head_activity_ci_heatmap,
    head_k_correlation_plot,
)
from src.latex_utils import comparisons_to_latex


def get_head_activity_rate_for_sample(src_token, task_dir, sample_idx, num_heads, num_layers):
    """
    Collect head activity info across all paths for a single sample of a task.

    Returns:
        heads_info[layer][head] = {
            "rate_active": float,  -> count_active / count_total
            "count_active": int,
            "count_total": int
        }
    """

    all_paths_by_src_token = {}
    src_token_path_data = torch.load(f"{task_dir}/{sample_idx}/{src_token}.pt", weights_only=False)

    paths = [src_token_path_data.loc[i]["path"] for i in range(len(src_token_path_data))]
    all_paths_by_src_token[src_token] = paths

    # Store raw counts first
    heads_info = {
        layer: {f"head_{i}": {"active": 0, "inactive": 0} for i in range(num_heads)} for layer in range(num_layers)
    }

    # Count activity
    for path in paths:
        for subtuple in path:
            if subtuple[1] is None:
                continue

            layer = subtuple[0]
            head_tensor = subtuple[1][0]  # boolean tensor of shape [num_heads]

            for head_idx in range(num_heads):
                if head_tensor[head_idx]:
                    heads_info[layer][f"head_{head_idx}"]["active"] += 1
                else:
                    heads_info[layer][f"head_{head_idx}"]["inactive"] += 1

    for layer in range(num_layers):
        for head_idx in range(num_heads):
            head_key = f"head_{head_idx}"
            count_active = heads_info[layer][head_key]["active"]
            count_inactive = heads_info[layer][head_key]["inactive"]

            count_total = count_active + count_inactive

            # Avoid division issues if something is empty
            if count_total == 0:
                heads_info[layer][head_key] = {
                    "mean": None,
                    "count_active": 0,
                    "count_total": 0,
                }
                continue

            rate_active = count_active / count_total

            heads_info[layer][head_key] = {
                "rate_active": float(rate_active),
                "count_active": count_active,
                "count_inactive": count_inactive,
                "count_total": count_total,
            }

    return heads_info


def bootstrap_head_activity(
        src_token, 
        task_dir, 
        samples, 
        num_heads, 
        num_layers, 
        n_bootstrap=10000,
    ):
    """
    Determine whether the head activity for a single sample is significanty different from noise.

    First collect per-sample activity rates for each (layer, head), compute bootstrapped
    CIs for the mean, then test whether the variance across samples is consistent
    with Gaussian noise via a chi-squared variance test.

    With N=100 samples (>30), the Central Limit Theorem justifies modelling the activity rates as i.i.d.
    Gaussian, making the chi-squared variance test exact. The null variance is
    p*(1-p) / mean(n_obs_i): the Gaussian approximation to the variance of a sample
    proportion, using the mean observation count as the effective sample size.
    A significant p-value means the head's variance across samples is too large or
    too small to be explained by noise.

    Args:
        src_token:   token position (loads {sample_idx}/{src_token}.pt per sample)
        task_dir:    directory containing per-sample subdirectories
        samples:     list of sample indices
        n_bootstrap: number of bootstrap resamples for CI estimation

    Returns:
        results[layer][head_key] = {
            "mean":       float        - mean activity rate across samples
            "ci_lower":   float        - bootstrap 95% CI lower bound on the mean
            "ci_upper":   float        - bootstrap 95% CI upper bound on the mean
            "rates":      list[float]  - per-sample activity rates
            "var_pvalue": float        - two-tailed chi-sq p-value vs Gaussian null
                                         (None if fewer than 2 valid samples or mean is 0/1)
        }
    """
    # per_sample[layer][head_key] = list of (rate, n_paths) one entry per valid sample
    per_sample = {
        layer: {f"head_{i}": [] for i in range(num_heads)}
        for layer in range(num_layers)
    }

    for sample_idx in samples:
        pt_path = f"{task_dir}/{sample_idx}/{src_token}.pt"
        try:
            src_token_path_data = torch.load(pt_path, weights_only=False)
        except FileNotFoundError:
            continue

        paths = [src_token_path_data.loc[i]["path"] for i in range(len(src_token_path_data))]
        if not paths:
            continue

        active_counts = {layer: {f"head_{i}": 0 for i in range(num_heads)} for layer in range(num_layers)}
        observations    = {layer: {f"head_{i}": 0 for i in range(num_heads)} for layer in range(num_layers)}

        for path in paths:
            for subtuple in path:
                if subtuple[1] is None:
                    continue
                layer = subtuple[0]
                head_tensor = subtuple[1][0]
                for head_idx in range(num_heads):
                    # Head is observed
                    head_key = f"head_{head_idx}"
                    observations[layer][head_key] += 1
                    if head_tensor[head_idx]:
                        # Head is active
                        active_counts[layer][head_key] += 1

        for layer in range(num_layers):
            for head_idx in range(num_heads):
                head_key = f"head_{head_idx}"
                num_observations = observations[layer][head_key]
                if num_observations > 0:
                    per_sample[layer][head_key].append((active_counts[layer][head_key] / num_observations, num_observations))

    rng = np.random.default_rng(seed=42)
    results = {}

    for layer in range(num_layers):
        results[layer] = {}
        for head_idx in range(num_heads):
            head_key = f"head_{head_idx}"
            entries = per_sample[layer][head_key]

            if not entries:
                results[layer][head_key] = {
                    "mean": None, "ci_lower": None, "ci_upper": None,
                    "rates": [], "var_pvalue": None,
                }
                continue

            active_rates = np.array([e[0] for e in entries])
            num_observations_array = np.array([e[1] for e in entries])
            N = len(active_rates)  # number of samples for which that head was observed ( = len(samples))
            mean_rate = float(active_rates.mean())

            # Bootstrap CI for the mean
            boot_means = np.array([
                rng.choice(active_rates, size=N, replace=True).mean()   # the mean of randomly-chosen activity rates
                for _ in range(n_bootstrap)
            ])
            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))

            # Chi-squared variance test against Gaussian null:
            # model r_i ~ N(p, σ²_null) where σ²_null = p*(1-p) / mean(n_obs_i)
            # (N-1)*S² / σ²_null ~ χ²(N-1) exactly for Gaussian data; valid here by Central Limit Theorem
            var_pvalue = None
            if N >= 2:
                null_var = mean_rate * (1 - mean_rate) / np.mean(num_observations_array)
                if null_var > 0:
                    S2 = float(np.var(active_rates, ddof=1))
                    stat = (N - 1) * S2 / null_var
                    p_lower = chi2_dist.cdf(stat, df=N - 1)
                    var_pvalue = float(2 * min(p_lower, 1 - p_lower))


            results[layer][head_key] = {
                "mean": mean_rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "rates": active_rates.tolist(),
                "var_pvalue": var_pvalue,
            }

    return results


def compare_heatmaps(heatmap_k1, heatmap_k2, num_heads, num_layers, activity_threshold=0.3):
    """
    Compare two mean activity rate heatmaps, e.g. from k=1 vs k=2.

    Each heatmap has the structure: heatmap[layer][head_key] = float (mean activity in [0, 1]),
    matching the format of tok_info[token_pos][task] after averaging over samples.

    Args:
        activity_threshold: heads with mean activity above this are considered "active"
                            for the Jaccard calculation.

    Returns:
        {
            "spearman_overall":   float         - Spearman r across all (layer, head) pairs
            "spearman_per_layer": list[float]   - per-layer Spearman r, length num_layers
            "spearman_per_head":  list[float]   - per-head Spearman r across layers, length num_heads
            "jaccard":            float         - Jaccard similarity of thresholded active-head masks
            "diff_heatmap":       np.ndarray    - shape (num_layers, num_heads), values are k2 - k1
        }
    """
    def to_matrix(heatmap):
        mat = np.zeros((num_layers, num_heads))
        for layer in range(num_layers):
            for head_idx in range(num_heads):
                val = heatmap[layer][f"head_{head_idx}"]
                mat[layer, head_idx] = val if val is not None else 0.0
        return mat

    mat1 = to_matrix(heatmap_k1)
    mat2 = to_matrix(heatmap_k2)

    r_overall, _ = spearmanr(mat1.flatten(), mat2.flatten())

    r_per_layer = []
    for layer in range(num_layers):
        r, _ = spearmanr(mat1[layer], mat2[layer])
        r_per_layer.append(float(r))

    r_per_head = []
    for head_idx in range(num_heads):
        r, _ = spearmanr(mat1[:, head_idx], mat2[:, head_idx])
        r_per_head.append(float(r))

    mask1 = mat1 > activity_threshold
    mask2 = mat2 > activity_threshold
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    jaccard = float(intersection / union) if union > 0 else 1.0

    return {
        "spearman_overall": float(r_overall),
        "spearman_per_layer": r_per_layer,
        "spearman_per_head": r_per_head,
        "jaccard": jaccard,
        "diff_heatmap": mat2 - mat1,
    }


def mean_activity_rate_per_token(k, task_pair, model_dir, samples, num_heads, num_layers):
    """
    Accumulate and average head activity info over all samples for a given k.

    Returns:
        tok_info:        defaultdict — tok_info[token_pos][task][layer][head_key] = mean activity rate
        prompt_tokens_1: DataFrame for the first task's prompt tokens (last valid sample)
        prompt_tokens_2: DataFrame for the second task's prompt tokens (last valid sample)
        len_longest:     length of the longest prompt seen across all samples
    """
    task_dirs = [
        f"{model_dir}/{task_pair[0]}/k={k}",
        f"{model_dir}/{task_pair[1]}/k={k}",
    ]

    tok_info = defaultdict(dict)
    len_longest = 0
    prompt_tokens_1 = None
    prompt_tokens_2 = None

    for sample_idx in tqdm(samples, desc=f"k={k}"):
        try:
            task_1 = os.listdir(f"{task_dirs[0]}/{sample_idx}")
            prompt_tokens_1 = pd.read_csv(f"{task_dirs[0]}/{sample_idx}/prompt_tokens.csv")
        except FileNotFoundError:
            continue
        try:
            task_2 = os.listdir(f"{task_dirs[1]}/{sample_idx}")
            prompt_tokens_2 = pd.read_csv(f"{task_dirs[1]}/{sample_idx}/prompt_tokens.csv")
        except FileNotFoundError:
            continue

        longest_prompt = max(len(prompt_tokens_1), len(prompt_tokens_2))
        if longest_prompt > len_longest:
            len_longest = longest_prompt

        for token_pos in range(longest_prompt):
            contribs_file = f"{token_pos}.pt"

            if contribs_file in task_1:
                heads_info = get_head_activity_rate_for_sample(token_pos, task_dirs[0], sample_idx, num_heads, num_layers)
                if task_pair[0] in tok_info[token_pos]:
                    for src_layer in tok_info[token_pos][task_pair[0]]:
                        for head in tok_info[token_pos][task_pair[0]][src_layer]:
                            # Take the head's mean activity
                            active_rate = heads_info[src_layer][head]["rate_active"]
                            if active_rate is not None:
                                tok_info[token_pos][task_pair[0]][src_layer][head] += active_rate
                else:
                    tok_info[token_pos][task_pair[0]] = {
                        layer: {head_key: (heads_info[layer][head_key]["rate_active"] or 0) for head_key in heads_info[layer]}
                        for layer in heads_info
                    }

            if contribs_file in task_2:
                heads_info = get_head_activity_rate_for_sample(token_pos, task_dirs[1], sample_idx, num_heads, num_layers)
                if task_pair[1] in tok_info[token_pos]:
                    for src_layer in tok_info[token_pos][task_pair[1]]:
                        for head in tok_info[token_pos][task_pair[1]][src_layer]:
                            # Take the head's mean activity
                            active_rate = heads_info[src_layer][head]["rate_active"]
                            if active_rate is not None:
                                tok_info[token_pos][task_pair[1]][src_layer][head] += active_rate
                else:
                    tok_info[token_pos][task_pair[1]] = {
                        layer: {head_key: (heads_info[layer][head_key]["rate_active"] or 0) for head_key in heads_info[layer]}
                        for layer in heads_info
                    }

    if prompt_tokens_1 is None or prompt_tokens_2 is None:
        raise RuntimeError(f"No data was found for k={k}.")

    for token_pos, task_dict in tok_info.items():
        for task in task_pair:
            if task not in task_dict:
                continue
            for src_layer, head_dict in task_dict[task].items():
                for head, mean_sum in head_dict.items():
                    tok_info[token_pos][task][src_layer][head] = round(mean_sum / len(samples), 4)

    return tok_info, prompt_tokens_1, prompt_tokens_2, len_longest


samples = [i for i in range(0, 100)]
task_idx = 0

# Choose a task pair
base_task = ["adjectives", "animals"][task_idx]
task_pair = [
    ["adjectives_adj_comp", "adjectives_adj_ant"], 
    ["animals_anim_color", "animals_can_fly"],
][task_idx]

query_tok_idx = 8
num_heads = 16
num_layers = 16

models = ["olmo-1b", "olmo-1b-sft", "olmo-1b-dpo"]
all_model_comparisons = {model: {} for model in models}

for model in models:
    model_dir = f"{PROJECT_ROOT}/experiments/output/path_analysis/{model}"

    tok_info_k1, _, _, _ = mean_activity_rate_per_token(
        k=1, task_pair=task_pair, model_dir=model_dir, samples=samples, num_heads=num_heads, num_layers=num_layers
    )
    tok_info_k2, _, _, _ = mean_activity_rate_per_token(
        k=2, task_pair=task_pair, model_dir=model_dir, samples=samples, num_heads=num_heads, num_layers=num_layers
    )
    comparisons = {}
    for task in task_pair:
        if (
            query_tok_idx in tok_info_k1 and task in tok_info_k1[query_tok_idx]
            and query_tok_idx in tok_info_k2 and task in tok_info_k2[query_tok_idx]
        ):
            comparisons[task] = compare_heatmaps(
                tok_info_k1[query_tok_idx][task],
                tok_info_k2[query_tok_idx][task],
                num_heads=num_heads,
                num_layers=num_layers,
            )
            all_model_comparisons[model][task] = comparisons[task]
    if comparisons:
        head_k_correlation_plot(
            comparisons,
            task_pair=[t for t in task_pair if t in comparisons],
            num_heads=num_heads,
            num_layers=num_layers,
            title=f"{model} | token_pos={query_tok_idx} | k=1 vs k=2 head correlation",
            save_path=f"{PROJECT_ROOT}/experiments/output/graphs/heads_{base_task}_{model}_k_correlation.png",
        )

    # Compute and plot bootstrapped confidence intervals
    bootstrap_src_tokens = {task_pair[0]: 8, task_pair[1]: 8}
    for bootstrap_k in [1, 2]:
        bootstrap_results = {
            task_pair[0]: bootstrap_head_activity(
                src_token=bootstrap_src_tokens[task_pair[0]],
                task_dir=f"{model_dir}/{task_pair[0]}/k={bootstrap_k}",
                samples=samples,
                num_heads=num_heads,
                num_layers=num_layers,
            ),
            task_pair[1]: bootstrap_head_activity(
                src_token=bootstrap_src_tokens[task_pair[1]],
                task_dir=f"{model_dir}/{task_pair[1]}/k={bootstrap_k}",
                samples=samples,
                num_heads=num_heads,
                num_layers=num_layers,
            ),
        }
        tok_label = ", ".join(f"{t}={bootstrap_src_tokens[t]}" for t in task_pair)
        head_activity_ci_heatmap(
            bootstrap_results,
            task_pair=task_pair,
            num_heads=num_heads,
            num_layers=num_layers,
            token_positions=bootstrap_src_tokens,
            title=f"token_pos: {tok_label} | k={bootstrap_k} | bootstrapped CIs + variance test",
            save_path=f"{PROJECT_ROOT}/experiments/output/graphs/heads_{base_task}_{model}_bootstrap_k{bootstrap_k}_10000.png",
        )

# Generate one LaTeX table per task, rows = models
for task in task_pair:
    task_comparisons = {
        model: all_model_comparisons[model][task]
        for model in models
        if task in all_model_comparisons[model]
    }
    if task_comparisons:
        comparisons_to_latex(
            task_comparisons,
            models=[m for m in models if m in task_comparisons],
            task=task,
            num_heads=num_heads,
            num_layers=num_layers,
            save_path=f"{PROJECT_ROOT}/experiments/output/tables/{base_task}_{task}_comparison.tex",
        )
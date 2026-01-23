"""
Script for conducting the t-test and creating the results tables.
"""

import math
import os

import torch
from scipy import stats

from arxiv2026_instruction_vectors import PROJECT_ROOT

def format_latex_pval_table(scores, t_stats, models, tasks, top_k):
    top_k_all_models = []

    # Format scores for latex table
    for model_idx in range(len(scores)):
        model = models[model_idx]
        t_stat = t_stats[model_idx]
        p_val = scores[model_idx]
        all_rows = ""

        for k_idx in range(top_k):
            kth_row_pattern = ""
            # Inefficient (we have to open and sort the same score files k times)
            for subtask_idx in range(len(tasks)): 
                t_stats_dict = t_stat[tasks[subtask_idx]]
                p_vals_dict = p_val[tasks[subtask_idx]]
                items_t = [item for item in t_stats_dict.items() if not math.isnan(item[1])]
                items_p = [item for item in p_vals_dict.items() if not math.isnan(item[1])]
                item_t = items_t[k_idx]
                item_p = items_p[k_idx]
                task_substr = f"{item_p[0]} & {'{:0.3e}'.format(item_t[1])} & {'{:0.3e}'.format(item_p[1])}"
                kth_row_pattern = kth_row_pattern + " & " + task_substr
            all_rows =  all_rows + kth_row_pattern + "  \\\\" + "\n"
        
        all_rows = all_rows.lstrip(" & ")
        all_rows = all_rows.replace("  \\\\\n & ","  \\\\\n")
        print(model)
        print(all_rows)
        top_k_all_models.append(items_t)
    return top_k_all_models

def load_and_merge_scores(dir, model, subtask, setting_name):
    path = f"{dir}/{model}/{subtask}"
    two_layer_file = f"{path}/{setting_name}-pos0.pt"
    scores_dict = torch.load(two_layer_file, weights_only=False)

    # If there are separate files for 1-LP scores, merge these into the 2LP dicts
    one_layer_file = f"{path}/1_layer_{setting_name}-pos0.pt"
    if os.path.exists(one_layer_file):
        one_layer_scores = torch.load(one_layer_file, weights_only=False)
        for key in one_layer_scores:
            # Basic check of the one-layer tuples
            if key[0] == key[1]:
                scores_dict[key] = one_layer_scores[key]
    return scores_dict

def statistical_analysis(unnorm_scores, norm_scores, tasks):
    top_k = 10

    t_stats_all_tasks = {}
    p_vals_all_tasks = {}
    alpha_mask_all_tasks = {}

    # Testing "f(L1)+f(L2) < f(L1UL2)"
    for subtask_idx in range(len(unnorm_scores)):

        alpha = 0.05

        t_stats = {}
        p_vals = {}
        alpha_mask = {}

        subtask_scores = unnorm_scores[subtask_idx]
        norm_subtask_scores = norm_scores[subtask_idx]
        
        # Use the normalized scores to get the top tuples
        items = [item for item in norm_subtask_scores.items() if not math.isnan(item[1])]
        items.sort(key=lambda item: item[1], reverse=True)
        top_items = [items[i] for i in range(top_k)]
        top_tuples = [item[0] for item in top_items]
        print(f"Top tuples for {tasks[subtask_idx]}:", top_tuples)

        sum_l1l2 = {}
        union_l1l2 = {}
        inequality_holds = {}

        # Gather unique layers involved in the top tuples
        for tuple in top_tuples:
            union_l1l2[tuple] = subtask_scores[tuple]
            inequality_holds[tuple] = (
                [
                    union_l1l2[tuple][i] >= 
                    ( subtask_scores[(tuple[0], tuple[0])][i] + 
                    subtask_scores[(tuple[1], tuple[1])][i] 
                    )
                    for i in range(100)   # calculate the sum for each i of the 100 samples
                ]
            )
            sum_l1l2[tuple] = (
                [
                    subtask_scores[(tuple[0], tuple[0])][i] + 
                    subtask_scores[(tuple[1], tuple[1])][i]
                    for i in range(100)   # calculate the sum for each i of the 100 samples
                ]
            )

        for tuple in top_tuples:
            t_stat, p_value = stats.ttest_1samp(inequality_holds[tuple], popmean=50.0)
            t_stats[tuple] = t_stat
            p_vals[tuple] = p_value
            alpha_mask[tuple] = "passed_test" if p_value < alpha else "failed_test"

        t_stats_all_tasks[tasks[subtask_idx]] = t_stats
        p_vals_all_tasks[tasks[subtask_idx]] = p_vals
        alpha_mask_all_tasks[tasks[subtask_idx]] = alpha_mask

    return t_stats_all_tasks, p_vals_all_tasks, alpha_mask_all_tasks


files_path = f"{PROJECT_ROOT}/experiments/output/patching_scores"

tasks = [
    "adj_ant",
    "adj_comp",
    "anim_color",
    "can_fly", 
]

task_names = [
    "adjective: antonym",
    "adjective: comparative",
    "animal: color",
    "animal: can_fly",
]

models = [
    "olmo-1b-dpo",
    "olmo-1b-sft",
    "olmo-1b",   
]

setting = [
    ["mean_logit_contribs_pt-to", "logits"], 
    ["mean_rank_pt-to", "rank"],
    ][1]

scores_m1 = [load_and_merge_scores(files_path, models[0], subtask, setting[0]) for subtask in tasks]
scores_m2 = [load_and_merge_scores(files_path, models[1], subtask, setting[0]) for subtask in tasks]
scores_m3 = [load_and_merge_scores(files_path, models[2], subtask, setting[0]) for subtask in tasks]


unnorm_scores_m1 = [load_and_merge_scores(files_path, models[0], subtask, "unnorm_logit_contribs_pt-to") for subtask in tasks]
unnorm_scores_m2 = [load_and_merge_scores(files_path, models[1], subtask, "unnorm_logit_contribs_pt-to") for subtask in tasks]
unnorm_scores_m3 = [load_and_merge_scores(files_path, models[2], subtask, "unnorm_logit_contribs_pt-to") for subtask in tasks]


unnorm_scores = [
    unnorm_scores_m1, 
    unnorm_scores_m2, 
    unnorm_scores_m3,
]

t_stats_all_tasks_1, p_vals_all_tasks_1, alpha_mask_all_tasks = statistical_analysis(unnorm_scores_m1, scores_m1, tasks)
t_stats_all_tasks_2, p_vals_all_tasks_2, alpha_mask_all_tasks = statistical_analysis(unnorm_scores_m2, scores_m2, tasks)
t_stats_all_tasks_3, p_vals_all_tasks_3, alpha_mask_all_tasks = statistical_analysis(unnorm_scores_m3, scores_m3, tasks)

scores = [
    p_vals_all_tasks_1,
    p_vals_all_tasks_2,
    p_vals_all_tasks_3
]

t_stats= [
    t_stats_all_tasks_1,
    t_stats_all_tasks_2,
    t_stats_all_tasks_3,
]

format_latex_pval_table(scores, t_stats, models, tasks, top_k=10)

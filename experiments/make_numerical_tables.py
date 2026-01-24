"""
Script for generating the numerical scores table for top_k tuples per task.
"""

from src.graph_utils import PROJECT_ROOT
import math
import os
import torch

def format_latex_scores(scores, models, top_k):
    top_k_all_models = []
    # Format scores for latex table
    for model_idx in range(len(scores)):
        model = models[model_idx]
        model_scores = scores[model_idx]
        all_rows = ""

        for k_idx in range(top_k):
            kth_row_pattern = ""
            # Inefficient (we have to open and sort the same score files k times)
            for subtask_idx in range(len(model_scores)):
                subtask_dict = model_scores[subtask_idx]
                #print("SUBTASK DICT:", subtask_dict.values())
                items = [item for item in subtask_dict.items() if not math.isnan(item[1])]
                items.sort(key=lambda item: item[1], reverse=True)

                item = items[k_idx]
                task_substr = f"{item[0]} & {item[1]}"
                kth_row_pattern = kth_row_pattern + " & " + task_substr
                #print("\n")
            all_rows =  all_rows + kth_row_pattern + "  \\\\" + "\n"
        
        all_rows = all_rows.lstrip(" & ")
        all_rows = all_rows.replace("  \\\\\n & ","  \\\\\n")
        print(model)
        print(all_rows)
        top_k_all_models.append(items)
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
    # 16 layers
    "olmo-1b-dpo",
    "olmo-1b-sft",
    "olmo-1b",
    
    # 32 layers
    #"olmo-7b-dpo",
    #"olmo-7b-sft",
    #"olmo-7b",
    
]

setting = [
    ["mean_logit_contribs_pt-to", "logits"], 
    ["mean_rank_pt-to", "rank"],
    ][1]


scores_m1 = [load_and_merge_scores(files_path, models[0], subtask, setting[0]) for subtask in tasks]
scores_m2 = [load_and_merge_scores(files_path, models[1], subtask, setting[0]) for subtask in tasks]
scores_m3 = [load_and_merge_scores(files_path, models[2], subtask, setting[0]) for subtask in tasks]


scores = [
    scores_m1, 
    scores_m2, 
    scores_m3,
]

top_tuples = format_latex_scores(scores, models, top_k=10)

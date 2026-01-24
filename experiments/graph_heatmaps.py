"""
Script for generating heatmaps from saved multilayer patching scores.
"""

from src.graph_utils import make_mlp_heatmap
from src import PROJECT_ROOT

import os

import torch


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

heatmap = make_mlp_heatmap(
        scores,
        save_path=f"{files_path}/{setting[1]}_pt-vs-to.pdf",
        num_layers=32,
        task_names=task_names,
        num_tasks=len(task_names),
        model_names=models,
        setting=setting[1],
        transpose_axis=False,
        colormap='viridis'
        )
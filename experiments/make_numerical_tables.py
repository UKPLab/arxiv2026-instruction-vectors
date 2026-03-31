"""
Script for generating the numerical scores table for top_k tuples per task.
"""

#from . import PROJECT_ROOT
import math
import os
import torch

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def format_latex_scores(scores, top_k, setting):
    top_k_all_models = []
    table_title = f"Contrastive Tasks - {setting} 1B"
    header = (
            r"\begin{table*}"
            "\n"
            r"\centering"
            "\n"
            r"\small"
            "\n"
            r"\begin{tabular}{cccccccc}"
            "\n"
            r"\toprule"
            "\n"
            + fr"\multicolumn{{8}}{{c}}{{\textbf{{{table_title}}}}} \\"
            "\n"
    )
    footer = (
        r"\bottomrule"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\caption{Caption}"
        "\n"
        r"\label{tab:app:label}"
        "\n"
        r"\end{table*}"
    )
    model_subtables = ""
    # Format scores for latex table
    for model_idx in range(len(scores)):
        model_scores = scores[model_idx]
        # User-specified variables for LaTeX header
        model_name = ["OLMo-2 1B SFT", "OLMo-2 1B DPO", "OLMo-2 1B"][model_idx]
        subtask_headers = ["adj: ant", "adj: comp", "anim: color", "anim: can\\_fly"]

        model_subtable_heading = (
            r"\toprule"
            + fr"\multicolumn{{8}}{{c}}{{\textbf{{{model_name}}}}} \\"
            "\n"
            r"\midrule"
            + " & ".join([fr"\multicolumn{{2}}{{c}}{{{header}}}" for header in subtask_headers]) + r"\\"
            "\n"
            r"\midrule"
            "\n"
            fr"Layer(s) &  {setting} contrib. & Layer(s) &  {setting} contrib. & Layer(s) &  {setting} contrib. &  Layer(s) &  {setting} contrib.\\"
            "\n"
            r"\midrule"
            "\n"
        )
        formatted_latex_rows = ""

        for k_idx in range(top_k):
            kth_row_pattern = ""
            # Inefficient (we have to open and sort the same score files k times)
            for subtask_idx in range(len(model_scores)):
                subtask_dict = model_scores[subtask_idx]
                items = [item for item in subtask_dict.items() if not math.isnan(item[1])]
                items.sort(key=lambda item: item[1], reverse=True)

                item = items[k_idx]
                task_substr = f"{item[0]} & {item[1]}"
                kth_row_pattern = kth_row_pattern + " & " + task_substr
                #print("\n")
            formatted_latex_rows =  formatted_latex_rows + kth_row_pattern + "  \\\\" + "\n"
        
        formatted_latex_rows = formatted_latex_rows.lstrip(" & ")
        formatted_latex_rows = formatted_latex_rows.replace("  \\\\\n & ","  \\\\\n")

        model_subtable = model_subtable_heading + formatted_latex_rows
        model_subtables = model_subtables + model_subtable
        top_k_all_models.append(items)

    complete_table_latex = header + model_subtables + footer
    return top_k_all_models, complete_table_latex


def load_and_merge_scores(dir, model, subtask, setting_name, num_patching_layers):
    path = f"{dir}/{model}/{subtask}"

    if num_patching_layers == 2:
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
                    
    elif num_patching_layers == 3:
        three_layer_file = f"{path}/3lp{setting_name}-pos0.pt"
        scores_dict = torch.load(three_layer_file, weights_only=False)

    
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
    "olmo-1b-sft",
    "olmo-1b-dpo",
    "olmo-1b",
    
    # 32 layers
    #"olmo-7b-sft",
    #"olmo-7b-dpo",
    #"olmo-7b",
    
]

setting = [
    ["mean_logit_contribs_pt-to", "Logits"], 
    ["mean_rank_pt-to", "Rank"],
    ][1]


scores_m1 = [load_and_merge_scores(files_path, models[0], subtask, setting[0], num_patching_layers=3) for subtask in tasks]
scores_m2 = [load_and_merge_scores(files_path, models[1], subtask, setting[0], num_patching_layers=3) for subtask in tasks]
scores_m3 = [load_and_merge_scores(files_path, models[2], subtask, setting[0], num_patching_layers=3) for subtask in tasks]


scores = [
    scores_m1, 
    scores_m2, 
    scores_m3,
]

top_tuples, latex_table = format_latex_scores(scores, top_k=10, setting=setting[1])

print(latex_table)

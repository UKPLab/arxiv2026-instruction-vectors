import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from src import PROJECT_ROOT, config
from src.graph_utils import (
    active_heads_heatmap_tinst_only,
)

def save_graphing_info(src_token, task_dir, sample_idx, num_heads):
    """
    Collect attention head activity info for a single sample of a task.

    A path is a tuple of "subtuples" that have the form:
    "((-1, None, 8), (0, (tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True]), True), 8)"

    Return heads_info, a dict containing the %activity and %inactivity for each head at each layer.
    The two values are trivially complementary - %inactivity can be inferred from %activity and vice versa.
    Both are included as a sanity check of correctness.
    
    """    
    all_paths_by_src_token = {}
    src_token_path_data = torch.load(f"{task_dir}/{sample_idx}/{src_token}.pt", weights_only=False)
    paths = [src_token_path_data.loc[i]["path"] for i in range(len(src_token_path_data))]
    all_paths_by_src_token[src_token] = paths
    
    heads_info = {
        layer: {f"head_{i}": [0, 0] for i in range(num_heads)}   # head_i: [#active, #inactive]
        for layer in range(config.n_layers)
    }

    for path_idx in range(len(paths)):
        # Read the subtuple structure of the path
        path = paths[path_idx]
        for j in range(len(path)):
            subtuple = path[j]
            if subtuple[1] is None:
                continue
            layer = subtuple[0]
            for head_idx in range(num_heads):
                head = subtuple[1][0][head_idx]            
                if head:
                    # head is active
                    heads_info[layer][f"head_{head_idx}"][0] += 1
                else:
                    # head is inactive
                    heads_info[layer][f"head_{head_idx}"][1] += 1

    # Calculate and store the activity/inactivity ratio for each head in each layer
    num_paths = len(paths)
    for layer in range(config.n_layers):
        for head_idx in range(num_heads):
            # Retrieve counts
            active_count = heads_info[layer][f"head_{head_idx}"][0]
            inactive_count = heads_info[layer][f"head_{head_idx}"][1]

            # Calculate ratio (as a percentage of paths), rounded to 2 decimals
            active_ratio = round(active_count / num_paths, 2)
            inactive_ratio = round(inactive_count / num_paths, 2)

            # Store back in heads_info
            heads_info[layer][f"head_{head_idx}"][0] = active_ratio
            heads_info[layer][f"head_{head_idx}"][1] = inactive_ratio

    return heads_info


model = "olmo-1b"

samples = [i for i in range(0, 100)]

task_idx = 0
model_dir = f"{PROJECT_ROOT}/experiments/output/analysis/path_analysis/{model}"
base_task = ["adjectives", "animals"][task_idx]
task_pair = [
    ["adjectives_adj_comp", "adjectives_adj_ant"],
    ["animals_anim_color", "animals_can_fly"]][task_idx]  # choose a task pair
task_dirs = [
    f"{PROJECT_ROOT}/experiments/output/analysis/path_analysis/{model}/{task_pair[0]}",
    f"{PROJECT_ROOT}/experiments/output/analysis/path_analysis/{model}/{task_pair[1]}"
    ]
query_tok_idx = 12
token_positions = list(range(20))
num_heads=16

tok_info = defaultdict(dict)
graph_file_s = f"{PROJECT_ROOT}/experiments/output/graphs/heads_{base_task}_avg_tinst.png"
len_longest_overall = 0

for sample_idx in tqdm(samples):
    # Only do those queries for which both complementary tasks were completed
    try:
        task_1 = os.listdir(f"{model_dir}/{task_pair[0]}/{sample_idx}")
        prompt_tokens_1 = pd.read_csv(f"{model_dir}/{task_pair[0]}/{sample_idx}/prompt_tokens.csv")
        decoded_1 = [prompt_tokens_1["decoded"][idx].replace("Ġ", "") for idx in range(len(prompt_tokens_1))]
    except FileNotFoundError:
        continue
    try:
        task_2 = os.listdir(f"{model_dir}/{task_pair[1]}/{sample_idx}")
        prompt_tokens_2 = pd.read_csv(f"{model_dir}/{task_pair[1]}/{sample_idx}/prompt_tokens.csv")
        decoded_2 = [prompt_tokens_2["decoded"][idx].replace("Ġ", "") for idx in range(len(prompt_tokens_2))]
    except FileNotFoundError:
        continue

    # Now for this query, check the output files for both tasks
    # Some token positions might have high-ranking paths in Task A, but not Task B - this is ok
    longest_prompt = max(len(prompt_tokens_1), len(prompt_tokens_2))

    if longest_prompt > len_longest_overall:
        len_longest_overall = longest_prompt

    for token_pos in range(longest_prompt):
        # This is the name of the paths file for this token (if the file exists)
        contribs_file = f"{token_pos}.pt"

        # Does this token_pos have a file in task 1?
        if contribs_file in task_1:
            task1_info, act_inact_ratios = save_graphing_info(token_pos, task_dirs[0], sample_idx, num_heads)
            if task_pair[0] in tok_info[token_pos]:
                for src_layer in tok_info[token_pos][task_pair[0]]:
                    for head in tok_info[token_pos][task_pair[0]][src_layer]:
                        tok_info[token_pos][task_pair[0]][src_layer][head][0] += act_inact_ratios[src_layer][head][0]
                        tok_info[token_pos][task_pair[0]][src_layer][head][1] += act_inact_ratios[src_layer][head][1]
            else:
                tok_info[token_pos][task_pair[0]] = act_inact_ratios

        # Does this token_pos have a file in task 2?
        if contribs_file in task_2:
            task2_info, act_inact_ratios = save_graphing_info(token_pos, task_dirs[1], sample_idx, num_heads)
            if task_pair[1] in tok_info[token_pos]:
                for src_layer in tok_info[token_pos][task_pair[1]]:
                    for head in tok_info[token_pos][task_pair[1]][src_layer]:
                        tok_info[token_pos][task_pair[1]][src_layer][head][0] += act_inact_ratios[src_layer][head][0]
                        tok_info[token_pos][task_pair[1]][src_layer][head][1] += act_inact_ratios[src_layer][head][1]
            else:
                tok_info[token_pos][task_pair[1]] = act_inact_ratios


# Average the head activity over all samples (each token position, task, layer, and head)
for token_pos, task_dict in tok_info.items():
    for task in task_pair:
        if task not in task_dict:
            continue
        for src_layer, head_dict in task_dict[task].items():
            for head, act_total in head_dict.items():
                # act_total is a list: [active_count, inactive_count]
                averaged = [round(count / len(samples), 2) for count in act_total]
                tok_info[token_pos][task][src_layer][head] = averaged


# Now, make the heatmap graph averaged over all token positions
active_heads_heatmap_tinst_only(
    tok_info,
    (task_pair[0], task_pair[1]), 
    len_longest_overall, 
    model,
    sample_idx, 
    num_heads=16, 
    num_layers=config.n_layers, 
    save_path=graph_file_s
)

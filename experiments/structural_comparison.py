import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from src import PROJECT_ROOT, config
from src.graph_utils import (
    active_heads_heatmap_tinst_only,
)

def save_graphing_info(task, src_token, task_dir, sample_idx, prompt_tokens, num_heads=16):
    all_paths_by_src = {}
    heads_df = defaultdict(list)
    heads_df["path_idx"]
    heads_df["path"]
    for i in range(num_heads):
        heads_df[f"head_{i}"]

    results_dir = f"{task_dir}/{sample_idx}"
    heads_file = f"{results_dir}/{src_token}_heads.csv"

    # Load the CSV containing the high-ranking paths
    contribs_df = torch.load(f"{task_dir}/{sample_idx}/{src_token}.pt", weights_only=False)
    paths = [contribs_df.loc[i]["path"] for i in range(len(contribs_df))]
    #tok_info[token_pos][task_pair[0]].append()

    paths = [contribs_df.loc[i]["path"] for i in range(len(contribs_df))]
    all_paths_by_src[src_token] = paths
    
    # Gather info on which heads are consistently active/inactive within 1 path
    activity = {layer: {f"head_{i}": [0, 0] for i in range(num_heads)} for layer in range(config.n_layers)}   # head_i: [%active, %inactive]

    
    for path_idx in range(len(paths)):
        path = paths[path_idx]
        for j in range(len(path)):
            subtuple = path[j]
            if subtuple[1] is None:
                continue
            layer = subtuple[0]
            for head_idx in range(num_heads):
                head = subtuple[1][0][head_idx]            
                if head:
                    activity[layer][f"head_{head_idx}"][0] += 1
                else:
                    activity[layer][f"head_{head_idx}"][1] += 1

    # Calculate the ratio of activity to inactivity for each layer-head across all subtuples&paths     
    for layer in range(config.n_layers):       
        for head_idx in range(num_heads):
            head_act_task1 = activity[layer][f"head_{head_idx}"][0]
            head_act_task2 = activity[layer][f"head_{head_idx}"][1]

            # Get rounded percentages
            head_act_task1 *= (1/len(paths))
            head_act_task2 *= (1/len(paths))
            head_act_task1 = round(head_act_task1, 2)
            head_act_task2 = round(head_act_task2, 2)

            activity[layer][f"head_{head_idx}"][0] = head_act_task1
            activity[layer][f"head_{head_idx}"][1] = head_act_task2

    heads_df = pd.DataFrame(heads_df)
    heads_df.to_csv(heads_file)
    return heads_df, activity


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

    query_1 = prompt_tokens_1.loc[query_tok_idx, "decoded"].replace("Ġ", "")
    query_2 = prompt_tokens_2.loc[query_tok_idx, "decoded"].replace("Ġ", "")

    # Now check the output files for both tasks this query.
    # Some token positions might have high-ranking paths in Task A, but not Task B - this is ok
    longest_prompt = max(len(prompt_tokens_1), len(prompt_tokens_2))

    if longest_prompt > len_longest_overall:
        len_longest_overall = longest_prompt

    for token_pos in range(longest_prompt):
        # This is the name of the paths file for this token (if the file exists)
        contribs_file = f"{token_pos}.pt"

        # Does this token_pos have a file in task 1?
        if contribs_file in task_1:
            task1_info, act_inact_ratios = save_graphing_info(task_pair[0], token_pos, task_dirs[0], sample_idx, prompt_tokens_1)
            if task_pair[0] in tok_info[token_pos]:
                for src_layer in tok_info[token_pos][task_pair[0]]:
                    for head in tok_info[token_pos][task_pair[0]][src_layer]:
                        tok_info[token_pos][task_pair[0]][src_layer][head][0] += act_inact_ratios[src_layer][head][0]
                        tok_info[token_pos][task_pair[0]][src_layer][head][1] += act_inact_ratios[src_layer][head][1]
            else:
                tok_info[token_pos][task_pair[0]] = act_inact_ratios

        # Does this token_pos have a file in task 2?
        if contribs_file in task_2:
            task2_info, act_inact_ratios = save_graphing_info(task_pair[1], token_pos, task_dirs[1], sample_idx, prompt_tokens_2)
            if task_pair[1] in tok_info[token_pos]:
                for src_layer in tok_info[token_pos][task_pair[1]]:
                    for head in tok_info[token_pos][task_pair[1]][src_layer]:
                        tok_info[token_pos][task_pair[1]][src_layer][head][0] += act_inact_ratios[src_layer][head][0]
                        tok_info[token_pos][task_pair[1]][src_layer][head][1] += act_inact_ratios[src_layer][head][1]
            else:
                tok_info[token_pos][task_pair[1]] = act_inact_ratios

# Average the head activity over all samples
for token_pos in tok_info:
    for task in task_pair:
        if task in tok_info[token_pos]:
            for src_layer in tok_info[token_pos][task]:
                for head in tok_info[token_pos][task][src_layer]:
                    act_total = tok_info[token_pos][task][src_layer][head]
                    act_perc = [round(t / len(samples), 2) for t in act_total]
                    tok_info[token_pos][task][src_layer][head] = act_perc


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

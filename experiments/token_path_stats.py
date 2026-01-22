"""
Script for graphing the high-rank path contributions per token position.
(Figure 4 of our paper)
"""

import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

size = "1b"
models = [f"olmo-{size}-dpo" , f"olmo-{size}-sft", f"olmo-{size}"]

samples = [i for i in range(0,100)]

tasks = [
    "adjectives_adj_comp", 
    "adjectives_adj_ant",
    "animals_anim_color", 
    "animals_can_fly"
]

t_inst_per_task = {
    "adjectives_adj_comp": 8, 
    "adjectives_adj_ant": 8,
    "animals_anim_color": 10, 
    "animals_can_fly": 19,
}

xlim_per_task = {
    "adjectives_adj_comp": (6, 17), 
    "adjectives_adj_ant": (6, 17),
    "animals_anim_color": (6, 18), 
    "animals_can_fly": (6, 28),
}

query_tok_idx = 12
num_tasks = 4
tok_info = defaultdict(dict)
graph_file_s = f"output/graphs/paths_count_{size}.png"
num_tokens = 30


# data per model
# 'task': (model_1, model_2, model_3)
model_data = {
    'adjectives_adj_comp': [],
    'adjectives_adj_ant': [],
    'animals_anim_color': [],
    'animals_can_fly': [],
}

for model in models:
    print("Model:", model)
    model_dir = f"output/path_analysis/{model}"

    for task in tasks:
        print("Task:", task)
        task_dirs = [
            f"{model_dir}/{task}",
            f"{model_dir}/{task}"
            ]
        
        num_completed_samples = 0
        token_stats_t1 = [0] * num_tokens

        for sample_idx in tqdm(samples):
            # Check that the folder isn't empty (i.e. the task was done for this sample)
            try:
                task_1 = os.listdir(f"{model_dir}/{task}/{sample_idx}")
                prompt_tokens_1 = pd.read_csv(f"{model_dir}/{task}/{sample_idx}/prompt_tokens.csv")
                decoded_1 = [prompt_tokens_1["decoded"][idx].replace("Ġ", "") for idx in range(len(prompt_tokens_1))]
                num_completed_samples += 1
            except FileNotFoundError:
                continue

            query_1 = prompt_tokens_1.loc[query_tok_idx, "decoded"].replace("Ġ", "")

            # Now check the output files for this sample
            # Some token positions might have high-ranking paths and others won't - this is ok
            for token_pos in range(num_tokens):

                # Does this token_pos have a file in the task?
                contribs_file = f"{model}_path_logit_contribs_{token_pos}.csv"
                if contribs_file in task_1:
                    df = pd.read_csv(f"{model_dir}/{task}/{sample_idx}/{contribs_file}")
                    num_paths = len(df)
                    token_stats_t1[token_pos] += num_paths

        # Normalize the sums according to the number of samples completed
        print("# Completed samples:", num_completed_samples)
        for tok_idx in range(len(token_stats_t1)):
            token_stats_t1[tok_idx] = token_stats_t1[tok_idx] / num_completed_samples

        model_data[task].append(token_stats_t1)
        

cmap = mpl.colormaps['cool']
colors = [
    "#00CA98",
    "#8095ff",
    "#99eae9",
]

x = np.arange(num_tokens)  # token positions
multiplier = 0

items = list(model_data.items())

fig = plt.figure(figsize=(3 * num_tasks, 5))
spec = fig.add_gridspec(ncols=3, nrows=2)

ax1 = fig.add_subplot(spec[1, :])
ax00 = fig.add_subplot(spec[0, 0], sharey=ax1)
ax01 = fig.add_subplot(spec[0, 1], sharey=ax1)
ax02 = fig.add_subplot(spec[0, 2], sharey=ax1)


axs = [ax00, ax01, ax02, ax1]

width = 0.8 / len(items[0][1])  # fit all models within one token slot

for ax, (task, models_scores) in zip(axs, items):
    x_lims = xlim_per_task[task]

    for model_idx, scores in enumerate(models_scores):
        offset = (model_idx - (len(models_scores) - 1) / 2) * width

        rects = ax.bar(
            x + offset,
            scores,
            width,
            label=f"{models[model_idx]}",
            color = colors[model_idx],
            linewidth=0.2, edgecolor="black"
        )
    
    x_lims = xlim_per_task[task]
    ax.set_xticks(x)
    ax.set_xlim(x_lims[0], x_lims[1])

    # Change the names of the first and last token (which are just there for visual purposes but have no data)
    ax.get_xticklabels()[t_inst_per_task[task]].set_color("blue")
    ax.get_xticklabels()[6].set_color("white")
    ax.get_xticklabels()[x_lims[1]].set_color("white")
    ax.set_xlabel("Token position")
    ax.set_yscale('log', base=10)
    ax.set_title(task)
    ax.legend(fontsize=8)
    

axs[0].set_ylabel("# of high-ranking paths")
axs[3].set_ylabel("# of high-ranking paths")
fig.tight_layout()
plt.savefig(graph_file_s, dpi=200)

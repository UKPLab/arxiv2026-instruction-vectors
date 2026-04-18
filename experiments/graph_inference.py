import os

import pandas as pd

from src import PROJECT_ROOT
from src.graph_utils import graph_inference_scores

score_path = f"{PROJECT_ROOT}/experiments/output/score_sheets/"

tasks = [
    "metaphor_boolean",
    "implicatures",
    "object_counting",
    "snarks",
    "animals_can_fly",
    "animals_anim_color",
    "adjectives_adj_comp",
    "adjectives_adj_ant",
]
metrics = ["basic_accuracy", "instruction_accuracy"]

model_groups = [
    ["olmo-1b", "olmo-1b-sft", "olmo-1b-dpo"],
    ["olmo-7b", "olmo-7b-sft", "olmo-7b-dpo"],
]
model_names = [m for group in model_groups for m in group]

summary = {m: pd.DataFrame(0.0, index=tasks, columns=model_names) for m in metrics}
for model in model_names:
    path = score_path + model + ".csv"
    if not os.path.exists(path):
        continue
    csv_cols = ["job", "model", "task", "num_samples", "basic_accuracy", "instruction_accuracy", "judge_accuracy"]
    df = pd.read_csv(path, names=csv_cols, skiprows=1)
    for _, row in df.iterrows():
        if row["task"] in tasks:
            instr_score = row["judge_accuracy"] if pd.notna(row["judge_accuracy"]) else row["instruction_accuracy"]
            summary["basic_accuracy"].loc[row["task"], model] = row["basic_accuracy"]
            summary["instruction_accuracy"].loc[row["task"], model] = instr_score

save_path = f"{PROJECT_ROOT}/experiments/output/graphs/inference_accuracies.pdf"
graph_inference_scores(
    summary["basic_accuracy"],
    summary["instruction_accuracy"],
    save_path,
    model_groups,
)

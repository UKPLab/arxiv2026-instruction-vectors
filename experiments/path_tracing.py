from src import PROJECT_ROOT, config
from src.data.load_datasets import load_task, InferenceDataset
from collections import defaultdict
import os
import sys
import torch
import gc
import pandas as pd

# Required by nnsight
sys.setrecursionlimit(10000)

# Full determinism for GPU calculations
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

from src.attn_tracer import get_attn_paths, get_model_for_paths, compute_path_transformation_vector
from src.utils import setup_model, get_token_position, print_gpu_tensors
from src.analysis.show_path_contributions import show_path_contributions
from src.analysis.plot_path_histogram import plot_path_histogram
from src.analysis.reduce_to_answer_contribution import reduce_to_answer_contribution
from src.analysis.filter_paths import filter_paths_by_threshold


# Save token data for graphing
def save_token_path_info(token, tokens_df, path_info, sample_idx, task, subtask=""):
    output_file = f"{config.short_name}_path_logit_contribs"
    tokens_df = pd.DataFrame.from_dict(tokens_df)

    all_contribs_s = {
        "path": [i[0]for i in path_info],
        "logit_contrib": [i[1]for i in path_info],
        "rank_of_target": [i[2]for i in path_info],
        }

    # Write logit contribs of each path to csv
    contribs_df = pd.DataFrame.from_dict(all_contribs_s)
    del all_contribs_s
    gc.collect()

    torch.save(contribs_df, f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/{token}.pt")
    contribs_df.to_csv(f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/{output_file}_{token}.csv")
    tokens_df.to_csv(f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/prompt_tokens.csv")
    print(f"Saved to file: {PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/{output_file}_{token}.csv")
    del contribs_df
    gc.collect()
    return

def run_subtask(model, tokenizer, task, subtask, sample_idx, start_pos, end_pos):
    # Load prompt (1 sample at a time)
    task_data = load_task(task, subtask, tokenizer)
    task_samples, task_instruction = (
        task_data.load_test(), 
        task_data.instruction_prompt
    )
    task_samples = InferenceDataset(task_samples)
    prompt, answer, _, _ = task_samples[sample_idx]
    print(f"Prompt:'{prompt}'\nAnswer:'{answer}'")

    # Make dirs for saving logit/rank contribution and graphs
    os.makedirs(f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}", exist_ok=True)
    hs_save_path = f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/pathwise_contribution_histograms.png"
    num_paths_save_path = f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}_{subtask}/{sample_idx}/total_paths_per_tok.csv"
    
    total_paths_token_positions = defaultdict(list)

    # Save prompt tokens
    tokens_df = {"token_id": [], "decoded": []}
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc_prompt["input_ids"][0].tolist())
    for tok_idx in range(len(tokens)):
        tokens_df["token_id"].append(tok_idx)
        tokens_df["decoded"].append(tokens[tok_idx])

    attn_paths = get_attn_paths(model, prompt, tokens, start_pos, end_pos)

    total_paths = sum(len(paths) for paths in attn_paths.values())
    print(f"Found {total_paths} paths from position {start_pos}")
    for src_token, paths in attn_paths.items():
        print(f"  Token {src_token} ('{tokens[src_token]}'): {len(paths)} paths")
        total_paths_token_positions[src_token].append(len(paths))

    print("\nCollecting model data for paths...")
    model_data = get_model_for_paths(model, prompt, attn_paths, start_pos)

    print_gpu_tensors()

    batch_size = 50_000
    print("\nComputing path transformations (pathwise with optimized layer-wise batching)...")
    transformations, path_contributions = compute_path_transformation_vector(
        attn_paths, model_data, config.model_base, batch_size=batch_size
    )

    path_data = reduce_to_answer_contribution(answer, path_contributions, model_data, tokenizer)
    plot_path_histogram(answer, path_data, hs_save_path)
    filtered_paths = filter_paths_by_threshold(path_data, threshold_rank=config.args.threshold_rank)

    pd.DataFrame.from_dict(total_paths_token_positions).to_csv(num_paths_save_path)
    print(f"Saved # total paths per token position to {num_paths_save_path}.")

    # Compute full transformation matrices for filtered paths
    print(f"\nComputing transformation matrices for {len(filtered_paths)} filtered paths...")

    # Convert filtered_paths back to the paths dictionary format expected by compute_path_transformation_matrix
    filtered_paths_dict = {}
    graph_info_dict = {}
    for src_token, path, ans_logit, ans_rank in filtered_paths:
        if src_token not in filtered_paths_dict:
            filtered_paths_dict[src_token] = []
            graph_info_dict[src_token] = []
        filtered_paths_dict[src_token].append(path)
        # Second dict for saving the info to a CSV
        graph_info_dict[src_token].append((path, ans_logit, ans_rank))
    
    return filtered_paths_dict, graph_info_dict, tokens_df


def main():
    sample_idx = config.args.tracing_sample_idx
    task = config.args.task
    subtask_dict = {
        "adjectives": ["adj_comp", "adj_ant"],
        "animals": ["anim_color", "can_fly"]
    }
    subtasks = subtask_dict[config.args.task]

    # Define start position per subtask
    start_positions = {
        "adj_comp": config.args.src_token_t1,
        "adj_ant": config.args.src_token_t2,
        "anim_color": config.args.src_token_t1,
        "can_fly": config.args.src_token_t2,
    }
    
    # Do both tasks for 1 sample
    filtered_paths_dicts = []
    for subtask in subtasks:

        model_path = config.model_name
        print("Model:", model_path)
        model, tokenizer = setup_model(model_path)

        # Start and end pos depend on subtask due to varying prompt lengths
        start_pos = start_positions[subtask]
        end_pos = config.args.end_pos

        print("Task:", task)
        print("Subtask:", subtask)
        print("Start Pos:", start_pos)
        filtered_paths_dict, graph_info_dict, tokens_df = run_subtask(model, tokenizer, task, subtask, sample_idx, start_pos, end_pos)
        filtered_paths_dicts.append(filtered_paths_dict)

        # Save token/path/answer rank info for graphing
        for src_token in graph_info_dict:
            save_token_path_info(src_token, tokens_df, graph_info_dict[src_token], sample_idx, task, subtask)

        del graph_info_dict
        gc.collect()

    print("Check len filtered paths dicts:", len(filtered_paths_dicts))
    # Save filtered paths dict
    torch.save(filtered_paths_dicts, f"{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}/{task}_sample_{sample_idx}_paths.pt")
    print(f"Saved filtered paths for subtask {subtask} to:{PROJECT_ROOT}/experiments/output/path_analysis/{config.short_name}/{task}/{task}_sample_{sample_idx}_paths.pt")

main()
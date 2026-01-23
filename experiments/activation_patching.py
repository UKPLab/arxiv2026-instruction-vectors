"""
Multilayer patching: Do combinatorial search for all combinations
of 2 or 3 layers. Then input to the patching function.
"""
import gc
import os
from itertools import combinations, product

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from nnsight import CONFIG, LanguageModel, util
from nnsight.tracing.graph import Proxy
from tqdm import tqdm

from arxiv2026_instruction_vectors import config, PROJECT_ROOT
from arxiv2026_instruction_vectors.data.load_datasets import load_task
from arxiv2026_instruction_vectors.metric_utils import compute_rr

load_dotenv()

login(token=os.getenv("API_TOKEN"))


# Make deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def combinatorial_search(num_choices, num_layers):
    # Example: [(1-to-1, 2-to-2), (2-to-2, 3-to-3), (1-to-1, 3-to-3)]
    # combinations = [(1, 2), (2, 3), (1, 3)]

    if num_choices == 1:
        tuples = [(l, l) for l in range(num_layers)]

    elif num_choices == 2:
        one_layer_tuples = [(l, l) for l in range(num_layers)]
        layers = [l for l in range(num_layers)]
        tuples = combinations(layers, num_choices)
        tuples = [t for t in tuples]
        tuples = tuples + one_layer_tuples

    elif num_choices == 3:
        layers = [l for l in range(num_layers)]
        tuples = combinations(layers, num_choices)
        tuples = [t for t in tuples]
        
    print("COMBINATIONS:", tuples)
    return tuples


def patch_multiple_layers(combinations):
    if config.args.subtask == "":
        subtask = config.args.task
    else:
        subtask = config.args.subtask

    # Make output dir:
    scores_dir = f"{PROJECT_ROOT}/experiments/output/patching_scores/{config.short_name}/{subtask}"
    os.makedirs(scores_dir, exist_ok=True)
    print("Made out dir:", scores_dir)

    # Load model
    model = LanguageModel(config.model_name, device_map="auto")
    print("MODEL:", model)

    # Load task
    task = load_task(given_name=config.args.task, subtask=config.args.subtask, tokenizer=model.tokenizer)

    cache_prompts, task_samples = (
        task.load_cache_prompts(), 
        task.load_test(), 
        task.instruction_prompt
    )

    cache_prompt = cache_prompts[0]

    # Prepare output paths for text responses and graphs
    os.makedirs(f"{PROJECT_ROOT}/experiments/output/graphs/{config.args.task}/{config.short_name}", exist_ok=True)
    os.makedirs(f"{PROJECT_ROOT}/experiments/output/responses/{config.args.task}/{config.short_name}", exist_ok=True)

    logit_diffs_all_samples_1 = {tuple: [] for tuple in combinations}
    logit_diffs_all_samples_2 = {tuple: [] for tuple in combinations}
    logit_diffs_all_samples_3 = {tuple: [] for tuple in combinations}
    logit_diffs_all_samples_4 = {tuple: [] for tuple in combinations}

    rr_diffs_all_samples_1 = {tuple: [] for tuple in combinations}
    rr_diffs_all_samples_2 = {tuple: [] for tuple in combinations}
    rr_diffs_all_samples_3 = {tuple: [] for tuple in combinations}

    for task_sample in tqdm(task_samples):
        instr_with_target = task_sample["instr+target"]
        target_prompt = task_sample["query"]   # default - just the target token
        #target_prompt = task_sample["prompt"]    # but maybe try "Adjective:'big'. Answer:"
        correct_token_id = model.tokenizer.encode(
            task_sample["mod_label"], add_special_tokens=False
            )[0] # remove enclosing list

        for targ_layer_tuple in combinations:
            logit_diffs_this_tuple_1 = []
            logit_diffs_this_tuple_2 = []
            logit_diffs_this_tuple_3 = []
            logit_diffs_this_tuple_4 = []
            rr_diffs_this_tuple_1 = []
            rr_diffs_this_tuple_2 = []
            rr_diffs_this_tuple_3 = []

            # Step 1: Run without patching but with instruction (Instruction + Target)
            with model.trace(instr_with_target) as tracer:
                if "qwen" in config.short_name:
                    instr_with_target_logits = model.lm_head.output.save()
                else:
                    instr_with_target_logits = model.output[0].save()
                instr_with_target_probs = torch.nn.functional.softmax(
                    instr_with_target_logits, dim=-1
                ).save()
            final_token_iwt = instr_with_target_probs[0, -1, :]
            rr_iwt, _ = compute_rr(final_token_iwt, correct_token_id, model, config.args.max_rank)

            # at the final instruction position, get the likeliest and 2nd-likeliest answer tokens
            highest_token_id = torch.argmax(final_token_iwt)
            decoded_1st = model.tokenizer.decode(highest_token_id)

            second_best_token_id = torch.topk(final_token_iwt, 2).indices[1]
            decoded_2nd = model.tokenizer.decode(second_best_token_id)

            # sanity check of sorting
            first_best_token_id = torch.topk(final_token_iwt, 2).indices[0]
            decoded_first = model.tokenizer.decode(first_best_token_id)

            del instr_with_target_probs
            gc.collect()

            # Step 2: Prepare patching: Do cache prompt run
            with model.trace(cache_prompt) as tracer:
                # Component names are architecture-specific
                if "qwen" in config.short_name:
                    cache_hs = {targ_layer:
                        model.model.layers[targ_layer].output[0].save() for targ_layer in targ_layer_tuple
                    }  # output shape (hidden_state_info,)
                else:
                    cache_hs = {targ_layer:
                        model.model.layers[targ_layer].output.save() for targ_layer in targ_layer_tuple
                      }  # output shape (hidden_state_info,)

            # Step 3: Run without patching or instructions (Target-only)
            with model.trace(target_prompt) as tracer:
                if "qwen" in config.short_name:
                    target_only_logits = model.lm_head.output.save()
                elif "gpt2" in config.short_name:
                    target_only_logits = model.output[0].save()
                else:
                    target_only_logits = model.output[0].save()
                target_only_probs = torch.nn.functional.softmax(
                    target_only_logits, dim=-1
                ).save()

            final_token_unm = target_only_probs[0, -1, :]
            rr_unm, _ = compute_rr(final_token_unm, correct_token_id, model, config.args.max_rank)

            # Step 4: Run with patching of cache_hs  (Patched + Target)
            # https://nnsight.net/notebooks/features/cross_prompt/
            source_pos = -1    # take the hidden state from the final token of the cache prompt
            target_positions = [0, -1]
            rr_per_targ_pos = {}
            for target_pos in target_positions:
                with model.trace(target_prompt) as tracer:
                    # Get unmodified logits, then do patching
                    if "llama" not in config.short_name:
                        for layer_idx in cache_hs.keys():
                            model.model.layers[layer_idx].output[:, target_pos, :] = (
                                cache_hs[layer_idx][:, source_pos, :]
                            )
                        patched_logits = model.output[0].save()
                    else:
                        # Target_pos 0 for llama models will be <|begin_of_text|>.
                        # We don't want to patch there, so increment by 1 to get the actual first token
                        if target_pos == 0:
                            llama_target_pos = 1
                        else:
                            llama_target_pos = target_pos
                        for layer_idx in cache_hs.keys():
                            model.model.layers[layer_idx].output[0][:, llama_target_pos, :] = (
                                cache_hs[layer_idx][:, source_pos, :]
                            )
                        patched_logits = model.output[0].save()
                
                patched_probs = torch.nn.functional.softmax(patched_logits, dim=-1)
                final_token = patched_probs[0, -1, :]
                rr_patched, _ = compute_rr(final_token, correct_token_id, model, max_rank=config.args.max_rank)
                rr_per_targ_pos[target_pos] = rr_patched

                # Calculate the effect of the patching, i.e. logit change of correct token at final output pos -1 (dim 1)
                # Patched+Target vs. Target-only
                patched_logit_diff_1 = (
                        patched_logits[0, -1, correct_token_id]
                        - target_only_logits[0, -1, correct_token_id]
                    )
                rr_diff_1 = (rr_patched - rr_unm)
                logit_diffs_this_tuple_1.append(patched_logit_diff_1.item())
                rr_diffs_this_tuple_1.append(rr_diff_1)

                # Patched+Target vs. Instruction+Target
                patched_logit_diff_2 = (
                    patched_logits[0, -1, correct_token_id]
                    - instr_with_target_logits[0, -1, correct_token_id]
                )
                rr_diff_2 = (rr_patched - rr_iwt)
                logit_diffs_this_tuple_2.append(patched_logit_diff_2.item())
                rr_diffs_this_tuple_2.append(rr_diff_2)

                # Additional Check: logit diff between highest-logit token and second-highest token
                # 1: patched setting
                patched_logit_diff_3 = (
                    patched_logits[0, -1, highest_token_id]
                    - patched_logits[0, -1, second_best_token_id]
                )
                logit_diffs_this_tuple_3.append(patched_logit_diff_3.item())
                # 2: target_only setting
                patched_logit_diff_4 = (
                    target_only_logits[0, -1, highest_token_id]
                    - target_only_logits[0, -1, second_best_token_id]
                )
                logit_diffs_this_tuple_4.append(patched_logit_diff_4.item())

                del patched_logits
                del patched_probs
                gc.collect()
                torch.cuda.empty_cache()
            
            logit_diffs_all_samples_1[targ_layer_tuple].append(logit_diffs_this_tuple_1)
            logit_diffs_all_samples_2[targ_layer_tuple].append(logit_diffs_this_tuple_2)
            logit_diffs_all_samples_3[targ_layer_tuple].append(logit_diffs_this_tuple_3)
            logit_diffs_all_samples_4[targ_layer_tuple].append(logit_diffs_this_tuple_4)

            rr_diffs_all_samples_1[targ_layer_tuple].append(rr_diffs_this_tuple_1)
            rr_diffs_all_samples_2[targ_layer_tuple].append(rr_diffs_this_tuple_2)
            rr_diffs_all_samples_3[targ_layer_tuple].append(rr_diffs_this_tuple_3)

            del logit_diffs_this_tuple_1
            del logit_diffs_this_tuple_2
            del logit_diffs_this_tuple_3
            del logit_diffs_this_tuple_4
            del instr_with_target_logits
            del cache_hs
            gc.collect()

        del target_only_logits
        gc.collect()

    # Get the Cartesian product of all the layers to define the graph grid
    # Some will be nan because we don't do these combinations, or they're duplicates
    # i.e. patching (0,1) == patching (1,0) but only (0,1) was done
    # while we don't patch (0,0) - this is equivalent to single-layer patching
    tuples = product([i for i in range(config.n_layers)], [i for i in range(config.n_layers)])
    tuples = [i for i in tuples]
    
    # Save all the logit and rank contribs to not have to rerun the patching
    # setting 1 (pt vs to), targ pos 0
    unnorm_scores_1 = {tuple: [scores[0] for scores in logit_diffs_all_samples_1[tuple]] # target_pos 0
                       if tuple in logit_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_1 = {tuple: np.round(np.mean(
                        [scores[0] for scores in logit_diffs_all_samples_1[tuple]] # target_pos 0
                        ), 2) 
                       if tuple in logit_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}

    unnorm_ranks_1 = {tuple:[scores[0] for scores in rr_diffs_all_samples_1[tuple]] # target_pos 0
                       if tuple in rr_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    mean_ranks_1 = {tuple: np.round(np.mean(
                        [scores[0] for scores in rr_diffs_all_samples_1[tuple]] # target_pos 0
                        ), 2) 
                       if tuple in rr_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    
    # setting 1 (pt vs to), targ pos 1
    unnorm_scores_2 = {tuple: [scores[1] for scores in logit_diffs_all_samples_1[tuple]] # target_pos 1
                       if tuple in logit_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_2 = {tuple: np.round(np.mean(
                        [scores[1] for scores in logit_diffs_all_samples_1[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in logit_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    
    unnorm_ranks_2 = {tuple: [scores[1] for scores in rr_diffs_all_samples_1[tuple]] # target_pos 1
                       if tuple in rr_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    mean_ranks_2 = {tuple: np.round(np.mean(
                        [scores[1] for scores in rr_diffs_all_samples_1[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in rr_diffs_all_samples_1.keys() else np.nan 
                       for tuple in tuples}
    
    # setting 2 (pt vs it), targ pos 0
    unnorm_scores_3 = {tuple: [scores[0] for scores in logit_diffs_all_samples_2[tuple]] # target_pos 0
                       if tuple in logit_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_3 = {tuple: np.round(np.mean(
                        [scores[0] for scores in logit_diffs_all_samples_2[tuple]] # target_pos 0
                        ), 2) 
                       if tuple in logit_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    
    unnorm_ranks_3 = {tuple: [scores[0] for scores in rr_diffs_all_samples_2[tuple]] # target_pos 0
                       if tuple in rr_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    mean_ranks_3 = {tuple: np.round(np.mean(
                        [scores[0] for scores in rr_diffs_all_samples_2[tuple]] # target_pos 0
                        ), 2) 
                       if tuple in rr_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    
    # setting 2 (pt vs it), targ pos 1
    unnorm_scores_4 = {tuple: [scores[1] for scores in logit_diffs_all_samples_2[tuple]] # target_pos 1
                       if tuple in logit_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_4 = {tuple: np.round(np.mean(
                        [scores[1] for scores in logit_diffs_all_samples_2[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in logit_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    
    unnorm_ranks_4 = {tuple: [scores[1] for scores in rr_diffs_all_samples_2[tuple]] # target_pos 1
                       if tuple in rr_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    mean_ranks_4 = {tuple: np.round(np.mean(
                        [scores[1] for scores in rr_diffs_all_samples_2[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in rr_diffs_all_samples_2.keys() else np.nan 
                       for tuple in tuples}
    

    # Save the top-k control checks
    unnorm_scores_5 = {tuple: [scores[1] for scores in logit_diffs_all_samples_3[tuple]] # target_pos 1
                       if tuple in logit_diffs_all_samples_3.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_5 = {tuple: np.round(np.mean(
                        [scores[1] for scores in logit_diffs_all_samples_3[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in logit_diffs_all_samples_3.keys() else np.nan 
                       for tuple in tuples}
    
    unnorm_scores_6 = {tuple: [scores[1] for scores in logit_diffs_all_samples_4[tuple]] # target_pos 1
                       if tuple in logit_diffs_all_samples_4.keys() else np.nan 
                       for tuple in tuples}
    mean_scores_6 = {tuple: np.round(np.mean(
                        [scores[1] for scores in logit_diffs_all_samples_4[tuple]] # target_pos 1
                        ), 2) 
                       if tuple in logit_diffs_all_samples_4.keys() else np.nan 
                       for tuple in tuples}


    if config.args.num_choices == 1:
        one_layer = "1_layer_"
    else:
        one_layer = ""

    if config.args.num_choices == 3:
        three_layers = "3lp"
    else:
        three_layers = ""

    # Save normalized scores
    torch.save(mean_scores_1, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_pt-to-pos0.pt")
    torch.save(mean_scores_2, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_pt-to-pos1.pt")
    torch.save(mean_scores_3, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_pt-it-pos0.pt")
    torch.save(mean_scores_4, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_pt-it-pos1.pt")
    torch.save(mean_scores_5, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_top2test_pt.pt")
    torch.save(mean_scores_6, f"{scores_dir}/{one_layer}{three_layers}mean_logit_contribs_top2test_to.pt")

    torch.save(mean_ranks_1, f"{scores_dir}/{one_layer}{three_layers}mean_rank_pt-to-pos0.pt")
    torch.save(mean_ranks_2, f"{scores_dir}/{one_layer}{three_layers}mean_rank_pt-to-pos1.pt")
    torch.save(mean_ranks_3, f"{scores_dir}/{one_layer}{three_layers}mean_rank_pt-it-pos0.pt")
    torch.save(mean_ranks_4, f"{scores_dir}/{one_layer}{three_layers}mean_rank_pt-it-pos1.pt")

    # Save unnormalized scores
    torch.save(unnorm_scores_1, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_pt-to-pos0.pt")
    torch.save(unnorm_scores_2, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_pt-to-pos1.pt")
    torch.save(unnorm_scores_3, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_pt-it-pos0.pt")
    torch.save(unnorm_scores_4, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_pt-it-pos1.pt")
    torch.save(unnorm_scores_5, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_top2test_pt.pt")
    torch.save(unnorm_scores_6, f"{scores_dir}/{one_layer}{three_layers}unnorm_logit_contribs_top2test_to.pt")

    torch.save(unnorm_ranks_1, f"{scores_dir}/{one_layer}{three_layers}unnorm_rank_pt-to-pos0.pt")
    torch.save(unnorm_ranks_2, f"{scores_dir}/{one_layer}{three_layers}unnorm_rank_pt-to-pos1.pt")
    torch.save(unnorm_ranks_3, f"{scores_dir}/{one_layer}{three_layers}unnorm_rank_pt-it-pos0.pt")
    torch.save(unnorm_ranks_4, f"{scores_dir}/{one_layer}{three_layers}unnorm_rank_pt-it-pos1.pt")

    print("Done.")
    print(f"Saved files to {scores_dir}.")


def main():
    combinations = combinatorial_search(config.args.num_choices, config.n_layers)
    patch_multiple_layers(combinations)

if __name__ == "__main__":
    main()
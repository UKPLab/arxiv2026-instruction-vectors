import numpy as np
import sklearn
import torch
from openai import OpenAI


# Compute rr of a single sample, given token_probs of final generation token pos
def compute_rr(token_probs, correct_token_id, model, max_rank=100):
    sorted_probs = torch.sort(token_probs, dim=-1, descending=True)
    cands_to_probs = {}
    for i in range(max_rank):
        cands_to_probs[sorted_probs.indices[i].item()] = sorted_probs.values[i].item()
    top_candidates = list(cands_to_probs.keys())

    # See if the correct token is among these, and if so, get its rank
    # Use 1-based indexing for computational purposes
    try:
        correct_token_rank = top_candidates.index(correct_token_id) + 1
    except ValueError:
        correct_token_rank = -1

    # Take reciprocal of the rank
    if correct_token_rank != 0:
        recip_rank = 1 / correct_token_rank
    else:
        recip_rank = -1                         # if the correct token isn't within the max_rank range
    return recip_rank, correct_token_rank


def compute_mrr(output_probs, correct_token_id, model, max_rank=10):

    # Sort the output probs from highest to lowest
    # This is a tensor of [indices, values]
    # Shape of each component: [1, token_pos, vocab_dim]
    sorted_probs = torch.sort(output_probs, dim=-1, descending=True)

    # Create dictionary of {tokens:probs}, len=max_rank for candidate tokens
    # take from last pos logit only
    # mrr is calculated across different samples

    # mean ( unm(sample1) - mod(sample1) + unm(sample2) - mod(sample2) )
    cands_to_probs = {}
    pairs = [(sorted_probs.indices[0, token_pos, :], sorted_probs.values[0, token_pos, :]) for token_pos in range(sorted_probs.indices.shape[1])]
    for indices, values in pairs:
        for i in range(max_rank):
            cands_to_probs[indices[i].item()] = values[i].item()
    ranked_tokens = sorted_probs.indices

    # Find the rank of the correct token separately (since this might be >> max_rank)
    corr_token_info = []
    for pos_info in pairs:
        indices = pos_info[0]
        probs = pos_info[1]
        corr_token_rank = list(indices).index(correct_token_id)
        corr_token_prob = probs[corr_token_rank].item()
        corr_token_info.append((corr_token_rank, corr_token_prob))

    # Calculate reciprocal ranks for the first n=max_rank tokens
    # Also get corresponding probs and decode
    recip_ranks = []

    for token_pos in range(ranked_tokens.shape[1]):
        top_candidates = ranked_tokens[0, token_pos, :][0:max_rank]
        cands_with_probs = [(cand.item(), cands_to_probs[cand.item()]) for cand in top_candidates]
        decoded_cands = [model.tokenizer.decode(t) for t in top_candidates]

        # See if the correct token is among these, and if so, get its rank
        # Use 1-based indexing for computational purposes
        try:
            correct_token_rank = list(top_candidates).index(correct_token_id) + 1
        except ValueError:
            correct_token_rank = 0

        # Compute mean reciprocal rank
        if correct_token_rank != 0:
            recip_rank = 1 / correct_token_rank
        else:
            recip_rank = 0
        recip_ranks.append(recip_rank)

    mrr = np.mean(recip_ranks)
    return mrr, cands_with_probs, decoded_cands, corr_token_info


def basic_accuracy(pred_batches, targs_batches):
    all_preds = []
    all_targs = []
    for batch_idx in range(len(pred_batches)):
        preds = pred_batches[batch_idx]
        targs = targs_batches[batch_idx]
        for pred in preds:
            all_preds.append(pred.strip())
        for targ in targs:
            all_targs.append(targ.strip())
    return 100*sklearn.metrics.accuracy_score(all_targs, all_preds)

def judge_accuracy(batched_outputs, batched_instructions, task):
    judge_prompts = {
        "adj_comp": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is an English adjective in the comparative form. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "adj_ant": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is an English adjective in its standard declarative form. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "anim_color": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is the name of a color. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "metaphor_boolean": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is either 'Y' or 'N'. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "implicatures": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is either 'Y' or 'N'. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "object_counting": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is either a positive integer or zero. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
        "snarks": "You will be given a task instruction and an answer someone wrote in response. Regardless of whether the answer is correct, determine whether the answer is either 'A' or 'B'. Output either 'yes' or 'no'. Instruction: '{}' Answer: '{}'. Your decision:",
    }
    judge_prompt = judge_prompts[task]
    decisions = []

    # Set up API client
    client = OpenAI()

    for batch_idx in range(len(batched_outputs)):
        output_batch = batched_outputs[batch_idx]
        instr_batch = batched_instructions[batch_idx]
        for idx in range(len(output_batch)):
            output = output_batch[idx]
            instr = instr_batch[idx]
            prompt = judge_prompt.format(instr, output)
            decision = client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
            decisions.append(decision.output_text)
            print("Response:", output)
            print("Decision:", decision.output_text)

    followed = decisions.count("yes")
    judge_acc = followed / len(decisions)
    return 100*judge_acc

def instruction_accuracy(pred_batches, opts_batches):
    followed_instrs = 0
    total_samples = sum([len(b) for b in pred_batches])
    for batch_idx in range(len(pred_batches)):
        preds = pred_batches[batch_idx]
        opts = opts_batches[batch_idx]
        preds = [pred.strip() for pred in preds]
        for pred in preds:
            if pred in opts:
                followed_instrs += 1
    return 100*followed_instrs/total_samples
import torch

# Compute rr of a single sample, given token_probs of final generation token pos
def compute_rr(token_probs, correct_token_id, max_rank=1000):
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

def reduce_to_answer_contribution(answer_str, path_contributions, model_data, tokenizer):
    answer_tokens = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
    if len(answer_tokens) == 0:
        print(f"Warning: Could not tokenize answer '{answer_str}'")
        return []

    answer_token_id = answer_tokens[0]
    answer_token_str = tokenizer.convert_ids_to_tokens([answer_token_id])[0]
    print(f"\nReducing {len(path_contributions)} paths to (path, (logit, rank)) tuples for answer token: '{answer_token_str}' (ID: {answer_token_id})")

    s_final_all = model_data['s_final_all']
    lm_head_w = model_data['lm_head_w']

    if s_final_all is None or lm_head_w is None:
        print("Warning: Missing model components for analysis")
        return []

    s_final_last = s_final_all[0, :].contiguous()

    # Create (path, (logit, rank)) tuples
    path_data = []

    for src_token, path, path_vector in path_contributions:
        y_pred_head_in = s_final_last * path_vector
        pred_logits = torch.matmul(lm_head_w, y_pred_head_in)
        answer_logit = pred_logits[answer_token_id].item()
        sorted_indices = torch.argsort(pred_logits, descending=True)
        answer_rank = (sorted_indices == answer_token_id).nonzero(as_tuple=True)[0].item() + 1

        path_data.append((src_token, path, answer_logit, answer_rank))

    return path_data
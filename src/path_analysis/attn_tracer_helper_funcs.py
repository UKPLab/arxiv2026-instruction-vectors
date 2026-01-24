import os
import sys
import src.config as config
from collections import defaultdict
from src.data.load_datasets import load_task, InferenceDataset

# Required by nnsight
sys.setrecursionlimit(10000)

# help with determinism
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
import numpy as np
import pandas as pd

torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

def main(subtask):

    # Load model
    model_path = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lm = LanguageModel(model_path, tokenizer=tokenizer, attn_implementation="eager", device_map="auto")

    # Load prompt (1 sample at a time)
    task = load_task(config.args.task, subtask, tokenizer)
    task_samples, task_instruction = (
        task.load_test(), 
        task.instruction_prompt
    )
    task_samples = InferenceDataset(task_samples)
    prompt, target, _ = task_samples[config.args.tracing_sample_idx]

    torch.cuda.empty_cache()
    num_layers = lm.model.config.num_hidden_layers


    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(prompt, return_tensors="pt")
    tokens        = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    seq_len       = len(tokens)
    last_token_ix = seq_len - 1
    last_layer    = num_layers - 1

    # Collect everything we need from the forward pass
    with lm.trace(prompt, output_attentions=True, return_dict=True) as tr:
        if "olmo" in config.short_name:
            X_dev = lm.model.embed_tokens.output[0].save()  # (T,H)
            pa_in = [lm.model.layers[L].post_attention_layernorm.input[0].save() for L in range(num_layers)]
            ff_in  = [lm.model.layers[L].post_feedforward_layernorm.input[0].save() for L in range(num_layers)]
            mlp_in = [lm.model.layers[L].mlp.input[0].save() for L in range(num_layers)]
            mlp_gate = [lm.model.layers[L].mlp.gate_proj.output.save() for L in range(num_layers)]
            final_layer_norm = lm.model.norm.input[0].save()  # (T,H)
            #l0 = lm.model.norm.output[0].save()
            true_logits = lm.lm_head.output[0][-1].save()
            attentions = lm.output.attentions.save()

        # Need to check for these models, does there exist a path without layer normalization? for olmo yes
        elif "llama" in config.short_name:
            pass
        elif "qwen" in config.short_name:
            X_dev = lm.model.embed_tokens.output[0].save()
            pa_in = [lm.model.layers[L].post_attention_layernorm.input[0].save() for L in range(num_layers)]

    # The answer
    answer = torch.argmax(true_logits)

    # Collect tokens with the greatest attention weight per layer
    max_jump = [attn[0].argmax(dim=2).detach().cpu() for attn in attentions]

    # (Un)embedding matrix and final norm weights
    lm_head_w = lm.lm_head.weight.detach()          # (V,H)
    fn_weight = lm.model.norm.weight.detach()       # (H,)
    Hsize  = pa_in[0].shape[1]

    # Build graph and collect paths
    merged_graph = defaultdict(list)  
    _tmp = defaultdict(lambda: defaultdict(set))

    for L in range(num_layers):
        Hh, T = max_jump[L].shape
        for h in range(Hh):
            for q in range(T):
                src_tok = int(max_jump[L][h, q])
                _tmp[(L - 1, src_tok)][q].add(h)

    layers = lm.model.layers

    for source_layer in range(-1, num_layers - 1):  # source_layer = -1 (embeddings) up to last-1
        target_layer = source_layer + 1
        Hn = layers[target_layer].self_attn.config.num_attention_heads
        for src_tok in range(seq_len):
            targets    = _tmp.get((source_layer, src_tok), {})
            edge_list  = []
            added_self = False
            for next_tok, heads in sorted(targets.items(), key=lambda kv: kv[0]):
                include_skip = (next_tok == src_tok)
                if include_skip:
                    added_self = True

                mask = torch.zeros(Hn, dtype=torch.bool)
                if heads:
                    idx = torch.tensor(sorted(heads), dtype=torch.long)
                    mask[idx] = True

                edge_list.append((next_tok, mask, include_skip))

            if not added_self:
                mask = torch.zeros(Hn, dtype=torch.bool)
                edge_list.append((src_tok, mask, True))
            merged_graph[(source_layer, src_tok)] = edge_list

    def enumerate_paths_from(src_tok: int):
        paths_s = []
        stack = [(-1, src_tok, [(-1, None, src_tok)])]
        while stack:
            layer_i, tok_idx, path_so_far = stack.pop()
            if layer_i == last_layer:
                if tok_idx == last_token_ix:
                    paths_s.append(tuple(path_so_far))
                continue
            for next_tok, mask, include_skip in reversed(merged_graph.get((layer_i, tok_idx), [])):
                stack.append((layer_i + 1, next_tok,
                            path_so_far + [(layer_i + 1, (mask, include_skip), next_tok)]))
        return paths_s

    start = 0
    tokens_df = {"token_id": [], "decoded": []}
    all_paths_by_src = {}
    total_paths = 0
    for s in range(start, seq_len):
        ps = enumerate_paths_from(s)
        all_paths_by_src[s] = ps
        total_paths += len(ps)

    print(f"Total complete paths across source tokens {start}..{seq_len-1}: {total_paths}")
    for s in range(start, seq_len):
        print(f"  src {s:2d} ('{tokens[s]}'): {len(all_paths_by_src[s])} paths")
        tokens_df["token_id"].append(s)
        tokens_df["decoded"].append(tokens[s]) 

    if total_paths == 0:
        raise RuntimeError("No complete paths found from ANY source token to the last token.")

    del attentions

    # rms_fold
    def rms_scales(weight: torch.Tensor, x_TxH: torch.Tensor) -> torch.Tensor:
        scale = (x_TxH.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        return scale * weight  # (T,H)


    s_attn = []
    s_ff   = []
    mlp_a  = []
    mlp_a_input_dep = []

    # build linearized variants of non-linearities
    for L in range(num_layers):
        layer = layers[L]

        s_attn.append(rms_scales(layer.post_attention_layernorm.weight.detach(), pa_in[L]))
        s_ff.append(  rms_scales(layer.post_feedforward_layernorm.weight.detach(), ff_in[L]))

        gate_W = layer.mlp.gate_proj.weight.detach()
        a_all_id = layer.mlp.act_fn(F.linear(mlp_in[L], gate_W))  # (T, M)
        a_all = layer.mlp.act_fn(mlp_gate[L])  # (T, M)
        mlp_a.append(a_all)
        mlp_a_input_dep.append(a_all_id)

    s_final_all  = rms_scales(fn_weight, final_layer_norm)  # (T,H)
    s_final_last = s_final_all[last_token_ix, :].contiguous()               # (H,)
    return layers, s_attn, mlp_a, s_ff, s_final_last, lm.model.config.hidden_size


@torch.inference_mode()
def apply_OV_heads(layer_idx: int, mask_kh: torch.Tensor, layers, hid_size) -> torch.Tensor:

    layer = layers[layer_idx]
    attn  = layer.self_attn

    head_dim  = attn.head_dim   #  128
    num_heads = attn.config.num_attention_heads  # 16
    num_kv    = getattr(attn.config, "num_key_value_heads", num_heads)

    vW = attn.v_proj.weight.detach()                 # (H,Hn*D)  (2048, 16*128)
    oW = attn.o_proj.weight.detach()                 # (H, Hn*D)   (config.num_attention_heads * self.head_dim, config.hidden_size)

    # Expand vW to get a 3D matrix as needed
    if True:#"olmo" in config.short_name:
        vW_3d = vW.view(num_heads, head_dim, hid_size)   # (16, 128, 2048)
        oW_3d = oW.view(num_heads, head_dim, hid_size)

    V_heads = vW_3d #vW.view(num_heads, head_dim, lm.model.config.hidden_size)               # (Hn, D, H)
    O_heads = oW.view(hid_size, num_heads, head_dim).permute(1, 0, 2).contiguous()  # (Hn, H, D)

    ov_transformation = torch.einsum("nhd,ndj->nhj", O_heads.to(device="cuda"),V_heads.to(device="cuda"))  # V_heads * O_heads
    full_OV_transform = torch.einsum('N, N H J -> H J', mask_kh.float().to(device="cuda"), ov_transformation.to(device="cuda"))

    return full_OV_transform.to(device="cpu")  # shape V_heads * O_heads

# Linearization of MLP
def apply_mlp_token(layer_idx, tok_idx, mlp_a, layers) -> torch.Tensor:
    layer  = layers[layer_idx]
    up_W   = layer.mlp.up_proj.weight.detach()
    down_W = layer.mlp.down_proj.weight.detach()


    # Shape of keys: mlp_transformations[(layer_idx, token_idx)]
    mlp_this_layer_token = mlp_a[(layer_idx, tok_idx)]
    temp = torch.einsum('... f, d f -> ... d f', mlp_this_layer_token.to(device="cuda"), down_W.to(device="cuda"))
    mlp_transformation = torch.einsum('... d f, f c -> ... d c', temp, up_W.to(device="cuda"))

    # NOTE: This assumes we're always doing one sample at a time, and that batch_dim (dim 0) is always 1
    return mlp_transformation[0, :, :].to(device="cpu")

def apply_edge_vec(layer_idx: int, 
                   target_token: int, 
                   heads_info, 
                   matrix: torch.Tensor, 
                   layers,
                   hidden_size,
                   mlp_transformations,
                   s_attn_all,
                   s_ff_all,
                   ) -> torch.Tensor:
    mask_kh, include_skip = heads_info

    # Attention
    ov_transformation = apply_OV_heads(layer_idx, mask_kh, layers, hidden_size)
    identity = torch.eye(hidden_size, hidden_size) # shape: (D, D)
    diag_s_attn_lyr_tok = torch.diag(s_attn_all[layer_idx][target_token])
    normalization_ov = diag_s_attn_lyr_tok 

    if "llama" not in config.short_name:
        component_1 = torch.mm(normalization_ov.to(device="cuda"), ov_transformation.to(device="cuda")) + identity.to(device="cuda") # diagonal matrix
    else:
        component_1 = torch.mm(normalization_ov, (ov_transformation + identity)) #normalization(ov_transformation + identity)   # diagonal matrix

    matrix = torch.mm(component_1, matrix.to(device="cuda")).to(device="cpu")

    # MLP
    mlp_transformation = mlp_transformations[(layer_idx, target_token)]
    diag_s_ff_lyr_tok = torch.diag(s_ff_all[layer_idx][target_token])
    normalization_mlp = diag_s_ff_lyr_tok

    if "llama" not in config.short_name:
        component_1_mlp = torch.mm(normalization_mlp.to(device="cuda"), mlp_transformation.to(device="cuda")) + identity.to(device="cuda")  #(normalization * mlp_transformation) + identity
    else:
        component_1_mlp = torch.mm(normalization_mlp, (mlp_transformation + identity))  #normalization * (mlp_transformation + identity)

    matrix = torch.mm(component_1_mlp, matrix.to(device="cuda")).to(device="cpu")

    return matrix

def apply_path_to_vec(path_tuple, subtask, hid_size) -> torch.Tensor:

    m = torch.eye(hid_size, hid_size)  # shape: (D, D)

    # expects a tuple of all the paths
    for i in range(1, len(path_tuple)):
        layer_i, heads_info, target_token = path_tuple[i]
        m = apply_edge_vec(layer_i, target_token, heads_info, m) 

    return m

# This function is applicable for olmo but not llama2
def apply_final_norm_linearized(vec: torch.Tensor, s_final_last) -> torch.Tensor:
    return s_final_last * vec

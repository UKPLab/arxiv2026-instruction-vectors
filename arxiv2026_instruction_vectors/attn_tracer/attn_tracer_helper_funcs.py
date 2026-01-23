import os
import sys
import arxiv2026_instruction_vectors.config as config

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
    if "olmo" in config.short_name:
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

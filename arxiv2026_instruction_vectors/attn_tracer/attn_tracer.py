import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig
import gc
from tqdm import tqdm

def get_attn_paths(model, prompt, prompt_tokens, start_pos, end_pos=-1):
    seq_len = len(prompt_tokens)
    if end_pos == -1:
        end_pos = seq_len
    num_layers = model.model.config.num_hidden_layers

    # Forward pass with attention tracking
    with model.trace(prompt, output_attentions=True, return_dict=True) as tr:
        attentions = model.output.attentions.save()

    # Extract attention argmax patterns
    max_jump = [attn[0].argmax(dim=2).cpu() for attn in attentions]

    # Build attention graph
    merged_graph = defaultdict(list)
    _tmp = defaultdict(lambda: defaultdict(set))

    for L in range(num_layers):
        Hh, T = max_jump[L].shape
        for h in range(Hh):
            for q in range(T):
                src_tok = int(max_jump[L][h, q].item())
                _tmp[(L - 1, src_tok)][q].add(h)

    # Create edge list for each (layer, token) pair
    for source_layer in range(-1, num_layers - 1):
        Hn = model.model.config.num_attention_heads
        for src_tok in range(seq_len):
            targets = _tmp.get((source_layer, src_tok), {})
            edge_list = []
            added_self = False

            for next_tok, heads in sorted(targets.items(), key=lambda kv: kv[0]):
                include_skip = next_tok == src_tok
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

    # Enumerate paths from start_pos to last token
    def enumerate_paths_from(src_tok):
        paths = []
        stack = [(-1, src_tok, [(-1, None, src_tok)])]
        last_layer = num_layers - 1
        last_token = seq_len - 1

        while stack:
            layer_i, tok_idx, path_so_far = stack.pop()

            if layer_i == last_layer:
                if tok_idx == last_token:
                    paths.append(tuple(path_so_far))
                continue

            for next_tok, mask, include_skip in reversed(merged_graph.get((layer_i, tok_idx), [])):
                stack.append(
                    (layer_i + 1, next_tok,
                     path_so_far + [(layer_i + 1, (mask, include_skip), next_tok)])
                )

        return paths

    # Get paths from start_pos onward
    all_paths = {}
    for s in range(start_pos, end_pos):
        all_paths[s] = enumerate_paths_from(s)
    return all_paths

def get_model_for_paths(model, prompt, paths, start_pos=None):
    config = model.model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    model_type = config.model_type

    #enc = model.tokenizer(prompt, return_tensors="pt")

    if start_pos is None:
        start_pos = min(paths.keys()) if paths else 0

    # Identify which (layer, token) pairs are accessed by paths
    accessed_positions = set()
    for source_token, token_paths in paths.items():
        for path in token_paths:
            for layer, mask_info, token_idx in path:
                if layer >= 0:
                    accessed_positions.add((layer, token_idx))

    # Collect activations during forward pass
    in_norm_inputs = []
    pa_norm_inputs = []
    ff_norm_inputs = []
    mlp_gate_outputs = []
    mlp_inputs = []

    # Get embeddings and final layer norm input
    with model.trace(prompt) as tr:
        X_embed = model.model.embed_tokens.output[0].save()
        final_norm_input = model.model.norm.input[0].save()
        true_logits = model.lm_head.output[0].save()

        for L in range(num_layers):
            layer_accessed = any(l == L for l, _ in accessed_positions)
            if not layer_accessed:
                continue

            # Llama uses pre-norm (input_layernorm), OLMo uses post-norm only
            if hasattr(model.model.layers[L], 'input_layernorm'):
                in_norm_inputs.append((L, model.model.layers[L].input_layernorm.input[0].save()))

            # Both have post_attention_layernorm
            pa_norm_inputs.append((L, model.model.layers[L].post_attention_layernorm.input[0].save()))

            # OLMo has post_feedforward_layernorm
            if hasattr(model.model.layers[L], 'post_feedforward_layernorm'):
                ff_norm_inputs.append((L, model.model.layers[L].post_feedforward_layernorm.input[0].save()))

            # For both OLMo and Llama, save gate_proj output for activation computation
            mlp_gate_outputs.append((L, model.model.layers[L].mlp.gate_proj.output.save()))

            # For OLMo, also save MLP input if needed
            if model_type != "llama":
                mlp_inputs.append((L, model.model.layers[L].mlp.input[0].save()))

    # Kill the trace object immediately after exiting context
    # Clear any intervention graph or internal state
    if hasattr(tr, '_graph'):
        del tr._graph
    if hasattr(tr, 'output'):
        del tr.output
    if hasattr(tr, '_saved_tensors'):
        tr._saved_tensors.clear()
    del tr

    # Force immediate cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Extract weights for accessed layers
    mlp_weights = {}
    attention_weights = {}
    layer_norm_weights = {}

    accessed_layers = set(layer_idx for layer_idx, _ in accessed_positions)

    for layer_idx in accessed_layers:
        layer = model.model.layers[layer_idx]

        # MLP weights needed for folding
        mlp_weights[layer_idx] = {
            'up_proj': layer.mlp.up_proj.weight.detach().clone().cuda(),
            'down_proj': layer.mlp.down_proj.weight.detach().clone().cuda(),
            'act_fn': layer.mlp.act_fn
        }
        # Only need gate_proj for non-llama models
        if model_type != "llama":
            mlp_weights[layer_idx]['gate_proj'] = layer.mlp.gate_proj.weight.detach().clone().cuda()

        # Attention projection weights
        attention_weights[layer_idx] = {
            'o_proj': layer.self_attn.o_proj.weight.detach().clone().cuda(),
            'v_proj': layer.self_attn.v_proj.weight.detach().clone().cuda(),
        }

        # Layer norm weights
        layer_norm_weights[layer_idx] = {
            'post_attention': layer.post_attention_layernorm.weight.detach().clone().cuda()
        }
        # Add input_layernorm if it exists (Llama has it, OLMo doesn't)
        if hasattr(layer, 'input_layernorm'):
            layer_norm_weights[layer_idx]['input'] = layer.input_layernorm.weight.detach().clone().cuda()
        # Add post_feedforward if it exists (OLMo has it, Llama doesn't)
        if hasattr(layer, 'post_feedforward_layernorm'):
            layer_norm_weights[layer_idx]['post_feedforward'] = layer.post_feedforward_layernorm.weight.detach().clone().cuda()

    # Convert proxy activations to GPU tensors with detach to break autograd graph
    X_embed_tensor = X_embed.detach().cuda()
    final_norm_tensor = final_norm_input.detach().cuda()
    true_logits_tensor = true_logits.detach().cuda()

    # Delete the original proxies to free references
    del X_embed
    del final_norm_input
    del true_logits

    in_norm_tensors = {}
    for layer_idx, proxy in in_norm_inputs:
        in_norm_tensors[layer_idx] = proxy.detach().cuda()
    del in_norm_inputs  # Delete proxy list

    pa_norm_tensors = {}
    for layer_idx, proxy in pa_norm_inputs:
        pa_norm_tensors[layer_idx] = proxy.detach().cuda()
    del pa_norm_inputs  # Delete proxy list

    ff_norm_tensors = {}
    for layer_idx, proxy in ff_norm_inputs:
        ff_norm_tensors[layer_idx] = proxy.detach().cuda()
    del ff_norm_inputs  # Delete proxy list

    mlp_gate_tensors = {}
    for layer_idx, proxy in mlp_gate_outputs:
        mlp_gate_tensors[layer_idx] = proxy.detach().cuda()
    del mlp_gate_outputs  # Delete proxy list

    mlp_input_tensors = {}
    for layer_idx, proxy in mlp_inputs:
        mlp_input_tensors[layer_idx] = proxy.detach().cuda()
    del mlp_inputs  # Delete proxy list

    # Get final layer norm weight and LM head weight before deleting model
    fn_weight = model.model.norm.weight.detach().clone().cuda() if hasattr(model.model, 'norm') else None
    lm_head_w = model.lm_head.weight.detach().clone().cuda() if hasattr(model, 'lm_head') else None

    # Delete entire model from GPU to save memory
    # First move the underlying model to CPU
    if hasattr(model, 'model'):
        model.model = model.model.cpu()

    # Delete the tokenizer if it exists
    if hasattr(model, 'tokenizer'):
        del model.tokenizer

    # Delete the entire LanguageModel wrapper
    del model

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()

    def rms_norm_linearization(weight, x):
        """RMS normalization linearization"""
        scale = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        return scale * weight

    # Compute linearized normalizations for all layers
    s_in_all = {}
    s_attn_all = {}
    s_ff_all = {}

    for layer_idx in accessed_layers:
        if layer_idx in in_norm_tensors and 'input' in layer_norm_weights[layer_idx]:
            s_in_all[layer_idx] = rms_norm_linearization(
                layer_norm_weights[layer_idx]['input'],
                in_norm_tensors[layer_idx]
            )

        if layer_idx in pa_norm_tensors and 'post_attention' in layer_norm_weights[layer_idx]:
            s_attn_all[layer_idx] = rms_norm_linearization(
                layer_norm_weights[layer_idx]['post_attention'],
                pa_norm_tensors[layer_idx]
            )

        if layer_idx in ff_norm_tensors and 'post_feedforward' in layer_norm_weights[layer_idx]:
            s_ff_all[layer_idx] = rms_norm_linearization(
                layer_norm_weights[layer_idx]['post_feedforward'],
                ff_norm_tensors[layer_idx]
            )

    # Final layer norm linearization
    s_final_all = rms_norm_linearization(fn_weight, final_norm_tensor) if fn_weight is not None else None

    # Compute linearized MLP transformations - folded into single matrix
    mlp_transformations = {}

    for layer_idx, token_idx in accessed_positions:
        if layer_idx not in mlp_weights:
            continue

        mlp = mlp_weights[layer_idx]

        # Both OLMo and Llama use gate_proj output for activation
        if layer_idx in mlp_gate_tensors:
            mlp_act = mlp['act_fn'](mlp_gate_tensors[layer_idx])
        else:
            continue

        if mlp_act.dim() == 3:
            mlp_act = mlp_act.squeeze(0)

        if token_idx < mlp_act.shape[0]:
            act_values = mlp_act[token_idx, :]

            # Fold the linearized MLP into a single transformation matrix
            # MLP computation: down_proj @ diag(act_fn(gate_proj @ x)) @ up_proj
            # Since act_values = act_fn(gate_proj @ x) at this specific input,
            # we compute: down_proj @ diag(act_values) @ up_proj

            # Step 1: Scale down_proj rows by activation values
            # down_proj shape: (hidden_size, mlp_dim)
            # act_values shape: (mlp_dim,)
            # Result: (hidden_size, mlp_dim) with each column scaled
            temp = torch.einsum("F, DF -> DF", act_values, mlp['down_proj'])

            # Step 2: Matrix multiply with up_proj
            # temp shape: (hidden_size, mlp_dim)
            # up_proj shape: (mlp_dim, hidden_size)
            # Result: (hidden_size, hidden_size)
            folded_mlp_matrix = torch.einsum("DF, FC -> DC", temp, mlp['up_proj'])

            # DO NOT bake in s_ff here - it will be applied during path transformation
            mlp_transformations[(layer_idx, token_idx)] = folded_mlp_matrix

            # Clean up the temporary weights immediately after use
            del temp

    # Clean up all unnecessary tensors from GPU before returning
    # Delete MLP weights - these are huge and not needed anymore
    del mlp_weights

    # Delete MLP activation tensors
    del mlp_gate_tensors
    del mlp_input_tensors

    # Delete intermediate norm tensors
    del in_norm_tensors
    del pa_norm_tensors
    del ff_norm_tensors
    del layer_norm_weights

    # Move attention weights to CPU since they're not immediately needed
    # They'll be moved back to GPU when actually used in compute_path_transformations
    for layer_idx in attention_weights:
        attention_weights[layer_idx]['o_proj'] = attention_weights[layer_idx]['o_proj'].cpu()
        attention_weights[layer_idx]['v_proj'] = attention_weights[layer_idx]['v_proj'].cpu()

    # Keep only needed slices
    X_embed_needed = X_embed_tensor[:max(paths.keys())+1] if paths else X_embed_tensor[:1]
    true_logits_needed = true_logits_tensor[-1:] if true_logits_tensor.dim() == 2 else true_logits_tensor
    lm_head_w_needed = lm_head_w if lm_head_w is not None else None
    s_final_needed = s_final_all[-1:] if s_final_all is not None else None

    # Delete full-size tensors
    del X_embed_tensor
    del final_norm_tensor
    del true_logits_tensor
    del lm_head_w
    del fn_weight

    # Keep only needed positions from s_*_all tensors
    s_in_needed = {}
    s_attn_needed = {}
    s_ff_needed = {}
    for layer_idx, token_idx in accessed_positions:
        if layer_idx in s_in_all:
            if layer_idx not in s_in_needed:
                s_in_needed[layer_idx] = {}
            s_in_needed[layer_idx][token_idx] = s_in_all[layer_idx][token_idx].clone()
        if layer_idx in s_attn_all:
            if layer_idx not in s_attn_needed:
                s_attn_needed[layer_idx] = {}
            s_attn_needed[layer_idx][token_idx] = s_attn_all[layer_idx][token_idx].clone()
        if layer_idx in s_ff_all:
            if layer_idx not in s_ff_needed:
                s_ff_needed[layer_idx] = {}
            s_ff_needed[layer_idx][token_idx] = s_ff_all[layer_idx][token_idx].clone()

    # Delete full s_*_all tensors
    del s_in_all
    del s_attn_all
    del s_ff_all
    del s_final_all

    # Force garbage collection and empty cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'mlp_transformations': mlp_transformations,
        'attention_weights': attention_weights,
        'hidden_size': hidden_size,
        'paths': paths,
        'accessed_positions': accessed_positions,
        's_in_all': s_in_needed,
        's_attn_all': s_attn_needed,
        's_ff_all': s_ff_needed,
        's_final_all': s_final_needed,
        'X_embed': X_embed_needed,
        'true_logits': true_logits_needed,
        'lm_head_w': lm_head_w_needed,
        'model_type': model_type
    }

def _compute_path_transformations_generic(paths, model_data, model_base, batch_size=5000, mode='vector'):
    mlp_transformations = model_data['mlp_transformations']
    attention_weights = model_data['attention_weights']
    hidden_size = model_data['hidden_size']
    s_in_all = model_data['s_in_all']
    s_attn_all = model_data['s_attn_all']
    s_ff_all = model_data.get('s_ff_all', {})
    s_final_all = model_data['s_final_all']
    X_embed = model_data['X_embed']
    true_logits = model_data['true_logits']
    lm_head_w = model_data['lm_head_w']
    model_type = model_data.get('model_type', model_base)  # i.e. 'olmo', 'llama'

    last_token_idx = None
    num_layers = 0
    for _, token_paths in paths.items():
        if token_paths:
            last_token_idx = token_paths[0][-1][2]
            num_layers = max(layer for layer, _, _ in token_paths[0] if layer >= 0) + 1 if token_paths[0] else 0
            break

    if last_token_idx is None:
        return {}, []

    s_final_last = s_final_all[0, :].contiguous() if s_final_all is not None else None

    path_list = []
    src_tokens = []
    for src_token, token_paths in paths.items():
        for path in token_paths:
            path_list.append(path)
            src_tokens.append(src_token)

    total_paths = len(path_list)
    print(f"\nProcessing {total_paths} paths layer-by-layer with batch size {batch_size}")

    if mode == 'vector':
        print(f"Initializing embeddings for {total_paths} paths...")
        path_data = torch.zeros((total_paths, hidden_size), device='cuda')

        # Group by source token for efficient embedding initialization
        unique_src = {}
        for i, src in enumerate(src_tokens):
            if src not in unique_src:
                unique_src[src] = []
            unique_src[src].append(i)

        for src, indices in unique_src.items():
            if src < X_embed.shape[0]:
                path_data[indices] = X_embed[src, :]
    elif mode == 'matrix':
        print(f"Initializing identity matrices for {total_paths} paths...")
        path_data = torch.eye(hidden_size, device='cuda').unsqueeze(0).expand(total_paths, -1, -1).clone()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    path_indices = [1] * total_paths 
    prev_tokens = src_tokens.copy()

    print("\nPre-computing transformation matrices...")

    pbar = tqdm(total=num_layers, desc="Processing layers", unit="layer")

    for current_layer in range(num_layers):
        active_mask = []
        active_indices = []
        layer_info = []  

        for i, (path, path_idx) in enumerate(zip(path_list, path_indices)):
            if path_idx < len(path):
                layer_idx, heads_info, target_token = path[path_idx]
                if layer_idx == current_layer:
                    active_mask.append(i)
                    active_indices.append(i)
                    layer_info.append((prev_tokens[i], target_token, heads_info))

        if not active_indices:
            pbar.update(1)
            continue

        num_active = len(active_indices)
        for batch_start in range(0, num_active, batch_size):
            batch_end = min(batch_start + batch_size, num_active)
            batch_indices = active_indices[batch_start:batch_end]
            batch_info = layer_info[batch_start:batch_end]

            batch_data = path_data[batch_indices]

            if current_layer in attention_weights:
                v_proj = attention_weights[current_layer]['v_proj']
                o_proj = attention_weights[current_layer]['o_proj']
                if not v_proj.is_cuda:
                    v_proj = v_proj.cuda()
                if not o_proj.is_cuda:
                    o_proj = o_proj.cuda()
                ov = torch.matmul(o_proj, v_proj)

                src_tgt_groups = {}
                for idx, (src, tgt, _) in enumerate(batch_info):
                    key = (src, tgt)
                    if key not in src_tgt_groups:
                        src_tgt_groups[key] = []
                    src_tgt_groups[key].append(idx)

                for (src, tgt), group_indices in src_tgt_groups.items():
                    group_data = batch_data[group_indices]

                    if model_type == "llama":
                        if current_layer in s_in_all and src in s_in_all[current_layer]:
                            s_in_vec = s_in_all[current_layer][src]
                            scaled_ov = ov * s_in_vec.unsqueeze(0)  # Broadcasting scales columns

                            if mode == 'vector':
                                ov_component = torch.matmul(scaled_ov, group_data.T).T
                            else:  # matrix
                                ov_component = torch.einsum('ij,bjk->bik', scaled_ov, group_data)
                        else:
                            if mode == 'vector':
                                ov_component = torch.matmul(ov, group_data.T).T
                            else:  # matrix
                                ov_component = torch.einsum('ij,bjk->bik', ov, group_data)
                    else: 
                        if current_layer in s_attn_all and tgt in s_attn_all[current_layer]:
                            s_attn_vec = s_attn_all[current_layer][tgt]
                            scaled_ov = ov * s_attn_vec.unsqueeze(1)  

                            if mode == 'vector':
                                ov_component = torch.matmul(scaled_ov, group_data.T).T
                            else:  # matrix
                                ov_component = torch.einsum('ij,bjk->bik', scaled_ov, group_data)
                        else:
                            if mode == 'vector':
                                ov_component = torch.matmul(ov, group_data.T).T
                            else:  # matrix
                                ov_component = torch.einsum('ij,bjk->bik', ov, group_data)

                    batch_data[group_indices] = group_data + ov_component

            tgt_groups = {}
            for idx, (src, tgt, _) in enumerate(batch_info):
                if tgt not in tgt_groups:
                    tgt_groups[tgt] = []
                tgt_groups[tgt].append(idx)

            for tgt, group_indices in tgt_groups.items():
                if (current_layer, tgt) not in mlp_transformations:
                    continue

                mlp = mlp_transformations[(current_layer, tgt)]
                group_data = batch_data[group_indices]

                if model_type == "llama":
                    if current_layer in s_attn_all and tgt in s_attn_all[current_layer]:
                        s_attn_vec = s_attn_all[current_layer][tgt]
                        scaled_mlp = mlp * s_attn_vec.unsqueeze(0) 

                        if mode == 'vector':
                            mlp_component = torch.matmul(scaled_mlp, group_data.T).T
                        else: 
                            mlp_component = torch.einsum('ij,bjk->bik', scaled_mlp, group_data)
                    else:
                        if mode == 'vector':
                            mlp_component = torch.matmul(mlp, group_data.T).T
                        else: 
                            mlp_component = torch.einsum('ij,bjk->bik', mlp, group_data)
                else: 
                    if current_layer in s_ff_all and tgt in s_ff_all[current_layer]:
                        s_ff_vec = s_ff_all[current_layer][tgt]
                        scaled_mlp = mlp * s_ff_vec.unsqueeze(1)  # Broadcasting scales rows

                        if mode == 'vector':
                            mlp_component = torch.matmul(scaled_mlp, group_data.T).T
                        else: 
                            mlp_component = torch.einsum('ij,bjk->bik', scaled_mlp, group_data)
                    else:
                        if mode == 'vector':
                            mlp_component = torch.matmul(mlp, group_data.T).T
                        else: 
                            mlp_component = torch.einsum('ij,bjk->bik', mlp, group_data)

                batch_data[group_indices] = group_data + mlp_component

            path_data[batch_indices] = batch_data

            for idx, (_, tgt, _) in enumerate(batch_info):
                global_idx = batch_indices[idx]
                prev_tokens[global_idx] = tgt
                path_indices[global_idx] += 1

            del batch_data

        pbar.update(1)
        pbar.set_postfix({'Active': num_active, 'GPU_MB': f'{torch.cuda.memory_allocated()/1024**2:.0f}'})


    pbar.close()

    print("\nAggregating results...")
    transformations = {}
    path_contributions = []

    if mode == 'vector':
        for i, src_token in enumerate(src_tokens):
            if src_token not in transformations:
                transformations[src_token] = torch.zeros(hidden_size, device='cuda')
            transformations[src_token] += path_data[i]
            path_contributions.append((src_token, path_list[i], path_data[i].clone()))

        approximation = sum(transformations.values()) if transformations else torch.zeros(hidden_size, device='cuda')

        if s_final_last is not None:
            y_pred_head_in = s_final_last * approximation
        else:
            y_pred_head_in = approximation

        if lm_head_w is not None:
            pred_logits = torch.matmul(lm_head_w, y_pred_head_in)
        else:
            pred_logits = None

        if true_logits is not None and pred_logits is not None:
            true_logits_last = true_logits[-1, :] if true_logits.dim() == 2 else true_logits

            cos_sim = F.cosine_similarity(
                pred_logits.float().unsqueeze(0),
                true_logits_last.float().unsqueeze(0),
                dim=1
            ).item()

            print(f"\nCosine similarity: {cos_sim:.6f}")

            explained_norm = (pred_logits.float().norm() / (true_logits_last.float().norm() + 1e-12)).item()
            print(f"Norm ratio (||predicted|| / ||true||): {explained_norm:.4f}")
    else:
        for i, src_token in enumerate(src_tokens):
            if src_token not in transformations:
                transformations[src_token] = torch.zeros((hidden_size, hidden_size), device='cuda')
            transformations[src_token] += path_data[i]
            path_contributions.append((src_token, path_list[i], path_data[i].clone()))

    gc.collect()
    torch.cuda.empty_cache()

    return transformations, path_contributions


def compute_path_transformation_vector(paths, model_data, model_base, batch_size=5000):
    return _compute_path_transformations_generic(paths, model_data, model_base, batch_size, mode='vector')


def compute_path_transformation_matrix(paths, model_data, model_base, batch_size=5000):
    transformations, _ = _compute_path_transformations_generic(paths, model_data, model_base, batch_size, mode='matrix')
    return transformations

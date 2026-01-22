import gc
import torch
import math
from typing import Tuple, Dict
import os
import sys

# Required by nnsight
sys.setrecursionlimit(10000)

# Full determinism for GPU calculations
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

from arxiv2026_instruction_vectors.attn_tracer.attn_tracer_helper_funcs import apply_edge_vec

def randomized_range_finder_batch(
    mats: torch.Tensor,
    rank: int,
    n_oversamples: int = 20,
    n_iter: int = 2,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    mats: (batch, n, n) -- input matrices
    rank: target rank r (not including oversamples)
    Returns: Q: (batch, n, r) orthonormal bases for range(mats)
    """
    if device is None:
        device = mats.device
    batch, n, _ = mats.shape
    l = min(n, rank + n_oversamples)
    # draw random gaussian test matrix Omega: shape (batch, n, l)
    Omega = torch.randn((batch, n, l), device=device, dtype=dtype)
    # Y = A @ Omega
    Y = mats.matmul(Omega)            # (batch, n, l)
    # power iterations to improve approximation (optional)
    for _ in range(n_iter):
        Y = mats.matmul(mats.transpose(-2, -1).matmul(Y))
    # QR to orthonormalize
    Q, _ = torch.linalg.qr(Y, mode='reduced')   # Q: (batch, n, l)
    # truncate to desired rank (numerical)
    r = min(rank, Q.shape[-1])
    Q = Q[:, :, :r]   # (batch, n, r)
    print("Q SHAPE:", Q.shape)
    return Q

# Transformation matrix similarity calculation
def get_similarity_between_paths (data_t1,
                                  data_t2,
                                  src_token_t1,
                                  src_token_t2,
                                  fig_savepath,
                                  model_base,
                                  layers,
                                  batch_size=5000,
                                  start_layer = 0,
                                  end_layer = None,
                                  last_normalize=False,
                                  target_rank = 50):
    # Get the transformation matrices for the comparison
    # Task A
    paths_t1 = data_t1[0]
    hidden_size_1 = data_t1[1]
    mlp_transformations_1 = data_t1[2]
    s_attn_all_1 = data_t1[3]
    s_ff_all_1 = data_t1[4]
    
    pathwise_transforms_A = []
    for path in paths_t1[src_token_t1]:
        matA = torch.eye(hidden_size_1, hidden_size_1)
        # Iterate over the subtuples
        for i in range(start_layer, end_layer):
            layer_i, heads_info, target_token_1 = path[i]
            if heads_info is None:
                continue
            # Slice out the subtuples that happen before the start_layer of interest, and apply_edge_vec for these
            matA = apply_edge_vec(
                layer_i, 
                target_token_1, 
                heads_info, 
                matA, 
                layers,
                hidden_size_1,
                mlp_transformations_1,
                s_attn_all_1,
                s_ff_all_1,
                ) 
        pathwise_transforms_A.append(matA)

    batchA = torch.stack(pathwise_transforms_A)
    print("BATCH A SHAPE:", batchA.shape)

    del paths_t1
    del hidden_size_1
    del mlp_transformations_1
    del s_attn_all_1
    del s_ff_all_1
    gc.collect()
    torch.cuda.empty_cache()

    # Task B
    paths_t2 = data_t2[0]
    hidden_size_2 = data_t2[1]
    mlp_transformations_2 = data_t2[2]
    s_attn_all_2 = data_t2[3]
    s_ff_all_2 = data_t2[4]

    pathwise_transforms_B = []

    for path in paths_t2[src_token_t2]:
        matB = torch.eye(hidden_size_2, hidden_size_2)
        # Iterate over the subtuples
        for i in range(start_layer, end_layer):
            layer_i, heads_info, target_token_2 = path[i]
            if heads_info is None:
                continue
            # Slice out the subtuples that happen before the start_layer of interest, and apply_edge_vec for these
            matB = apply_edge_vec(
                layer_i, 
                target_token_2, 
                heads_info, 
                matB, 
                layers,
                hidden_size_2,
                mlp_transformations_2,
                s_attn_all_2,
                s_ff_all_2,
                ) 
        pathwise_transforms_B.append(matB)

    batchB = torch.stack(pathwise_transforms_B)
    print("BATCH B SHAPE:", batchB.shape)

    del paths_t2
    del hidden_size_2
    del mlp_transformations_2
    del s_attn_all_2
    del s_ff_all_2
    gc.collect()
    torch.cuda.empty_cache()

    res = pairwise_subspace_similarities(
                batchA, batchB,
                rank=target_rank,
                oversamples=20,
                n_iter=2,
                device='cuda:0',
                dtype=torch.float32
    )
    return res

# Stable batched SVD helper
def stable_batched_svd(M: torch.Tensor, driver: str = 'gesvdj') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    M: (..., r, c)
    returns U, S, Vh with same batch shape as M[:-2]
    Uses try/except to fall back to a more robust driver if needed.
    """
    try:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    except RuntimeError:
        # retry in double with gesvdj
        M_d = M.double()
        U_d, S_d, Vh_d = torch.linalg.svd(M_d, full_matrices=False, driver='gesvdj')
        U, S, Vh = U_d.to(M.dtype), S_d.to(M.dtype), Vh_d.to(M.dtype)
    return U, S, Vh


# Pairwise subspace similarity between two sets of matrices
def pairwise_subspace_similarities(
    matsA: torch.Tensor,
    matsB: torch.Tensor,
    rank: int = 256,
    oversamples: int = 20,
    n_iter: int = 2,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    normalize_trace: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    matsA: (m, n, n)
    matsB: (p, n, n)
    Returns dictionary with:
      - 'norm_trace_sim': (m, p) similarity in [0,1] (trace normalized by min(rankA,rankB))
      - 'affinity_sim':   (m, p) sqrt(trace)/sqrt(min_rank) in [0,1]
      - 'max_angle':      (m, p) maximum principal angle in radians
      - 'geodesic':       (m, p) sqrt(sum theta^2)
      - 'chordal':        (m, p) sqrt(sum sin^2 theta)
      - 'singular_values': (m, p, r_min) singular values (cosines) for each pair (padded/truncated)
      - 'ranks': (m,), (p,) estimated ranks used (here both equal to chosen rank truncated by n)
    """
    if device is None:
        device = matsA.device
    matsA = matsA.to(device=device, dtype=dtype)
    matsB = matsB.to(device=device, dtype=dtype)

    m, n, _ = matsA.shape
    p, _, _ = matsB.shape  

    # Step 1: compute orthonormal bases QA, QB
    QA = randomized_range_finder_batch(matsA, rank=rank, n_oversamples=oversamples, n_iter=n_iter, device=device, dtype=dtype)  # (m,n,rA)
    QB = randomized_range_finder_batch(matsB, rank=rank, n_oversamples=oversamples, n_iter=n_iter, device=device, dtype=dtype)  # (p,n,rB)
    rA = QA.shape[-1]
    rB = QB.shape[-1]
    rmin = min(rA, rB)

    # Pre-allocate result tensors
    norm_trace_sim = torch.empty((m, p), device=device, dtype=dtype)
    affinity_sim = torch.empty((m, p), device=device, dtype=dtype)
    max_angle = torch.empty((m, p), device=device, dtype=dtype)
    geodesic = torch.empty((m, p), device=device, dtype=dtype)
    chordal = torch.empty((m, p), device=device, dtype=dtype)

    singular_values = torch.empty((m, p, rmin), device=device, dtype=dtype)

    eye_small = None

    for j in range(p):
        QA_perm = QA.permute(0, 2, 1)           # (m, rA, n)
        # result: (m, rA, rB)
        M_batch = QA_perm.matmul(QB[j])        # (m, rA, rB)

        M_batch = torch.nan_to_num(M_batch, nan=0.0, posinf=1e6, neginf=-1e6)
        if eye_small is None or eye_small.shape[-1] != M_batch.shape[-1]:
            eye_small = torch.eye(M_batch.shape[-1], device=device, dtype=dtype).expand(M_batch.shape[0], -1, -1)
        M_batch = M_batch + 1e-12 * eye_small

        # batched SVD on M_batch (shape (m, rA, rB))
        U_batch, S_batch, Vh_batch = stable_batched_svd(M_batch)  # S_batch: (m, min(rA,rB))

        # clamp singular values
        S_batch = torch.clamp(S_batch, -1.0, 1.0)

        # Save singular values
        if S_batch.shape[-1] >= rmin:
            singular_values[:, j, :] = S_batch[:, :rmin]
        else:
            # pad with zeros if needed (unlikely because rmin <= rA,rB)
            pad = torch.zeros((m, rmin - S_batch.shape[-1]), device=device, dtype=dtype)
            singular_values[:, j, :S_batch.shape[-1]] = S_batch
            singular_values[:, j, S_batch.shape[-1]:] = pad

        # compute derived metrics
        sig = S_batch  # (m, r_current)
        thetas = torch.acos(sig)     # principal angles (m, r_current)
        sinsq = 1.0 - sig**2         # (m, r_current)
        trace_PA_PB = torch.sum(sig**2, dim=-1)  # (m,)

        # normalize trace similarity by min(rankA, rankB) to map to [0,1]
        denom = float(min(rA, rB))
        norm_trace_sim[:, j] = trace_PA_PB / denom
        # affinity similarity normalized: sqrt(trace)/sqrt(min_rank)
        affinity_sim[:, j] = torch.sqrt(trace_PA_PB) / math.sqrt(denom)

        # distances
        chordal[:, j] = torch.sqrt(torch.clamp(torch.sum(sinsq, dim=-1), min=0.0))
        max_angle[:, j] = torch.max(thetas, dim=-1).values
        geodesic[:, j] = torch.sqrt(torch.clamp(torch.sum(thetas**2, dim=-1), min=0.0))

    results = {
        'norm_trace_sim': norm_trace_sim,    
        'affinity_sim': affinity_sim,        
        'max_angle': max_angle,              
        'geodesic': geodesic,                
        'chordal': chordal,                  
        'singular_values': singular_values,  
        'ranks': (rA, rB),
    }
    return results

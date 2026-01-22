
"""
Test file to verify that single token inputs produce 100% cosine similarity
for different model architectures (LLaMA, Gemma, OLMo).
"""

import os
import sys
import torch
import pytest
import gc

sys.setrecursionlimit(10000)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

from arxiv2026_instruction_vectors.attn_tracer import get_attn_paths, get_model_for_paths, compute_path_transformation_vector, compute_path_transformation_matrix
from arxiv2026_instruction_vectors.utils import setup_model


MODEL_CONFIGS = [
    ("./models", "OLMo", "olmo"),
    ("./llama-2-7b-chat/7B-Chat", "LLaMA-2-7B-Chat", "llama"),
    ("./google/gemma-7b", "Gemma-7B", "gemma"),
]


@pytest.fixture(params=MODEL_CONFIGS, ids=[config[1] for config in MODEL_CONFIGS])
def model_config(request):
    """Fixture that provides model configuration."""
    model_path, model_name, model_type = request.param

    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")

    return model_path, model_name, model_type


@pytest.fixture
def model_and_tokenizer(model_config):
    model_path, model_name, model_type = model_config

    print(f"\nLoading {model_name} from {model_path}")
    model, tokenizer = setup_model(model_path)

    yield model, tokenizer, model_name, model_type

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def get_test_prompts(model_type):
    if model_type == "llama":
        # LLaMA: empty string gives just BOS token
        return [("", "BOS token")]
    elif model_type == "gemma":
        # Gemma: empty string gives just BOS token (since it adds BOS by default)
        return [("", "BOS token")]
    else:
        # OLMo: doesn't add special tokens, so we can use single words
        return [
            ("The", "The"),
            ("Hello", "Hello"),
            ("A", "A"),
        ]


def test_single_token_cosine_similarity(model_and_tokenizer):
    model, tokenizer, model_name, model_type = model_and_tokenizer

    test_prompts = get_test_prompts(model_type)

    for prompt, description in test_prompts:
        print(f"\nTesting {model_name} with prompt: '{description}'")

        # Store tokenizer reference since get_model_for_paths deletes it
        if not hasattr(model, 'tokenizer'):
            model.tokenizer = tokenizer

        start_pos = 0
        attn_paths = get_attn_paths(model, prompt, start_pos)

        total_paths = sum(len(paths) for paths in attn_paths.values())
        print(f"  Found {total_paths} paths from position {start_pos}")

        model_data = get_model_for_paths(model, prompt, attn_paths, start_pos)

        batch_size = 5000
        transformations, path_contributions = compute_path_transformation_vector(
            attn_paths, model_data, batch_size=batch_size
        )

        approximation = sum(transformations.values()) if transformations else torch.zeros(model_data['hidden_size'], device='cuda')

        s_final_last = model_data['s_final_all'][0, :].contiguous() if model_data['s_final_all'] is not None else None
        if s_final_last is not None:
            y_pred_head_in = s_final_last * approximation
        else:
            y_pred_head_in = approximation

        if model_data['lm_head_w'] is not None:
            pred_logits = torch.matmul(model_data['lm_head_w'], y_pred_head_in)
        else:
            pred_logits = None

        true_logits = model_data['true_logits']
        if true_logits is not None and pred_logits is not None:
            true_logits_last = true_logits[-1, :] if true_logits.dim() == 2 else true_logits

            cos_sim = torch.nn.functional.cosine_similarity(
                pred_logits.float().unsqueeze(0),
                true_logits_last.float().unsqueeze(0),
                dim=1
            ).item()

            print(f"  Cosine similarity: {cos_sim:.10f}")
            print(f"  Batch size: {batch_size}")

            
            
            # For single token inputs, we expect very high cosine similarity
            # but due to numerical precision, it might not be exactly 1.0
            expected_threshold = 1e-3 if model_type == "gemma" else 1e-5
            assert abs(cos_sim - 1.0) < expected_threshold, \
                f"Cosine similarity {cos_sim:.10f} is not close to 1.0 for single token input"

            
            norm_ratio = (pred_logits.float().norm() / (true_logits_last.float().norm() + 1e-12)).item()
            print(f"  Norm ratio: {norm_ratio:.6f}")

            assert abs(norm_ratio - 1.0) < 0.01, \
                f"Norm ratio {norm_ratio:.6f} is not close to 1.0"

        
        del transformations
        del path_contributions
        del model_data
        gc.collect()
        torch.cuda.empty_cache()


def test_matrix_vector_equivalence(model_and_tokenizer):
    model, tokenizer, model_name, model_type = model_and_tokenizer

    test_prompts = get_test_prompts(model_type)

    for prompt, description in test_prompts:
        print(f"\nTesting matrix/vector equivalence for {model_name} with prompt: '{description}'")

        # Store tokenizer reference since get_model_for_paths deletes it
        if not hasattr(model, 'tokenizer'):
            model.tokenizer = tokenizer

        start_pos = 0
        attn_paths = get_attn_paths(model, prompt, start_pos)

        total_paths = sum(len(paths) for paths in attn_paths.values())
        print(f"  Found {total_paths} paths from position {start_pos}")

        
        model_data = get_model_for_paths(model, prompt, attn_paths, start_pos)

        
        batch_size = 5000

        
        transformations_vec, path_contributions_vec = compute_path_transformation_vector(
            attn_paths, model_data, batch_size=batch_size
        )

        
        transformations_mat, path_contributions_mat = compute_path_transformation_matrix(
            attn_paths, model_data, batch_size=batch_size
        )


        for key in transformations_vec.keys():
            vec_trans = transformations_vec[key]
            mat_trans = transformations_mat[key]

            # Get the embedding for this source token
            if key < model_data['X_embed'].shape[0]:
                embedding = model_data['X_embed'][key]
                # Apply the transformation matrix to the embedding
                mat_result = torch.matmul(mat_trans, embedding)

                # Check if the results are close
                if not torch.allclose(vec_trans, mat_result, rtol=1e-5, atol=1e-8):
                    max_diff = torch.max(torch.abs(vec_trans - mat_result)).item()
                    print(f"  WARNING: Transformation {key} differs by max {max_diff}")

                assert torch.allclose(vec_trans, mat_result, rtol=1e-5, atol=1e-8), \
                    f"Vector and matrix transformations differ for key {key}"

        
        assert len(path_contributions_vec) == len(path_contributions_mat), \
            "Different number of path contributions"

        for i, ((src_v, path_v, contrib_v), (src_m, path_m, contrib_m)) in enumerate(
            zip(path_contributions_vec, path_contributions_mat)
        ):
            assert src_v == src_m, f"Source tokens differ at index {i}: {src_v} vs {src_m}"
            assert path_v == path_m, f"Paths differ at index {i}: {path_v} vs {path_m}"

            # Apply the transformation matrix to the embedding for this source token
            if src_m < model_data['X_embed'].shape[0]:
                embedding = model_data['X_embed'][src_m]
                mat_contrib_result = torch.matmul(contrib_m, embedding)

                if not torch.allclose(contrib_v, mat_contrib_result, rtol=1e-5, atol=1e-8):
                    max_diff = torch.max(torch.abs(contrib_v - mat_contrib_result)).item()
                    print(f"  WARNING: Contribution {i} differs by max {max_diff}")

                assert torch.allclose(contrib_v, mat_contrib_result, rtol=1e-5, atol=1e-8), \
                    f"Contributions differ at index {i}"

        print("  ✓ Matrix and vector methods produce identical results")

        
        del transformations_vec, transformations_mat
        del path_contributions_vec, path_contributions_mat
        del model_data
        gc.collect()
        torch.cuda.empty_cache()



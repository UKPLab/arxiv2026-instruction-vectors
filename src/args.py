import argparse

def create_parser():
    parser = argparse.ArgumentParser(
                    description='Command line arguments.',
                )
    
    # General model and inference params
    parser.add_argument('--model_idx', type=int, default=19)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--subtask', type=str, default="")
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_rank', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=5)

    # Activation patching args
    parser.add_argument('--cache_prompt_idx', type=int, default=0)
    parser.add_argument('--num_choices', type=int, default=2, choices=[2, 3])
    
    # Start and end token positions for path tracing
    parser.add_argument('--start_pos', type=int, default=9)
    parser.add_argument('--end_pos', type=int, default=-1)

    # Dataset sample index for path tracing (iteration over samples is done in bash script)
    parser.add_argument('--tracing_sample_idx', type=int, default=0)

    # Linear probe and geometric analysis args 
    parser.add_argument('--varying', type=str, default="instructions", choices=["instructions", "labels"])
    parser.add_argument('--reduction_method', type=str, default="pca", choices=["pca", "tsne", "lda"])
    parser.add_argument('--model_component', type=str, default="resid_post", choices=["resid_post", "attn_out", "mlp"])
    parser.add_argument('--layer_for_dataset', type=int, default=0)
    parser.add_argument('--make_dataset', action="store_true")
    parser.add_argument('--do_probe', action="store_true")
    
    return parser
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
                    description='Command line arguments.',
                )
    parser.add_argument('--model_idx', type=int, default=19)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--subtask', type=str, default="")
    parser.add_argument('--cache_prompt_idx', type=int, default=0)
    parser.add_argument('--instruction_idx', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--max_rank', type=int, default=10000)
    parser.add_argument('--load_8bit', action="store_true")
    parser.add_argument('--load_4bit', action="store_true")
    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=5)
    parser.add_argument('--num_choices', type=int, default=3)
    parser.add_argument('--num_tuples', type=int, default=None)
    parser.add_argument('--do_tuples', type=tuple, default=None, action="append")
    parser.add_argument('--rewrite_existing', action="store_true")
    parser.add_argument('--local_data', action="store_true")
    parser.add_argument('--varying', type=str, default="labels")
    parser.add_argument('--reduction_method', type=str, default="pca")
    parser.add_argument('--model_component', type=str, default="resid_post")
    parser.add_argument('--start_token', type=int, default=4)
    parser.add_argument('--end_token', type=int, default=8)
    parser.add_argument('--tracing_sample_idx', type=int, default=0)
    parser.add_argument('--combos_to_do', type=str, default=None)
    parser.add_argument('--layer_for_dataset', type=int, default=0)
    parser.add_argument('--make_dataset', action="store_true")
    parser.add_argument('--do_probe', default=True)
    
    return parser
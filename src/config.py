"""
Specify model information.
"""
from src.args import create_parser

# Set project args
parser = create_parser()
args = parser.parse_args()

# Get model info
model_info = [# repo name, outfile name, num_layers
        ["openai-community/gpt2", "gpt2", 12],
        ["Qwen/Qwen2.5-0.5B", "qwen2.5-0.5b", 24, "qwen"],                              # 1
        ["Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct", 24, "qwen"],            # 2
        ["Qwen/Qwen2.5-3B", "qwen2.5-3b", 24, "qwen"],                                  # 3
        ["Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3b-instruct", 24, "qwen"],                # 4
        ["Qwen/Qwen2.5-7B", "qwen2.5-7b", 24, "qwen"],                                  # 5
        ["Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", 24, "qwen"],                # 6
        ["Qwen/Qwen2.5-32B", "qwen2.5-32b", 24, "qwen"],                                # 7
        ["Qwen/Qwen2.5-32B-Instruct", "qwen2.5-32b-instruct", 24, "qwen"],              # 8
        ["Qwen/Qwen2.5-72B", "qwen2.5-72b", 24, "qwen"],                                # 9
        ["Qwen/Qwen2.5-72B-Instruct", "qwen2.5-72b-instruct", 24, "qwen"],              # 10
        ["meta-llama/Meta-Llama-3-8B", "llama-3-8B", 32, "llama"],                       # 11
        ["meta-llama/Meta-Llama-3-8B-Instruct", "llama-3-8B-instruct", 32, "llama"],     # 12
        ["meta-llama/Meta-Llama-3-70B", "llama-3-70B", 32, "llama"],                     # 13
        ["meta-llama/Meta-Llama-3-70B-Instruct", "llama-3-70B-instruct", 32, "llama"],   # 14
        ["allenai/OLMo-2-0425-1B-SFT", "olmo-1b-sft", 16, "olmo"],   # 15
        ["allenai/OLMo-2-0425-1B-DPO", "olmo-1b-dpo", 16, "olmo"],   # 16
        ["allenai/OLMo-2-0425-1B", "olmo-1b", 16, "olmo"],           # 17
        ["allenai/OLMo-2-1124-7B-SFT", "olmo-7b-sft", 32, "olmo"],   # 18
        ["allenai/OLMo-2-1124-7B-DPO", "olmo-7b-dpo", 32, "olmo"],   # 19
        ["allenai/OLMo-2-1124-7B", "olmo-7b", 32, "olmo"],           # 20
        ["allenai/OLMo-2-1124-13B-SFT", "olmo-13b-sft", 40, "olmo"], # 21
        ["allenai/OLMo-2-1124-13B-DPO", "olmo-13b-dpo", 40, "olmo"], # 22
        ["allenai/OLMo-2-1124-13B", "olmo-13b", 40, "olmo"],         # 23
        ["allenai/OLMo-2-0325-32B-SFT", "olmo-32b-sft", 64, "olmo"], # 24
        ["allenai/Olmo-3-1025-7B", "olmo3-7b", 32, "olmo"],          # 25
        ["allenai/Olmo-3-7B-Instruct-SFT", "olmo3-7bi-sft", 32, "olmo"],  # 26
        ["allenai/Olmo-3-7B-Instruct-DPO", "olmo3-7bi-dpo", 32, "olmo"],  # 27
        ][args.model_idx]
model_name = model_info[0]
short_name = model_info[1]
n_layers = model_info[2]
model_base = model_info[3]
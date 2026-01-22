from transformers import AutoTokenizer, AutoConfig
from nnsight import LanguageModel
import torch
import gc
def setup_model(model_path):
    model_type = AutoConfig.from_pretrained(model_path).model_type
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token

    lm = LanguageModel(
        model_path, tokenizer=tokenizer, device_map="cuda:0", attn_implementation="eager"
    )
    return lm, tokenizer

def get_token_position(prompt: str, subword: str, tokenizer) -> int:
    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    sub_enc = tokenizer(subword, add_special_tokens=False, return_offsets_mapping=True)
    sub_tokens = tokenizer.convert_ids_to_tokens(sub_enc["input_ids"])

    n = len(sub_tokens)

    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == sub_tokens:
            return i
    return 0


def print_gpu_tensors():
    gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"\nGPU memory used: {gpu_memory_gb:.2f} GB")
    print("\nTensors on GPU:")
    tensor_groups = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                key = (obj.shape, obj.dtype)
                if key not in tensor_groups:
                    tensor_groups[key] = {'count': 0, 'size_mb': obj.element_size() * obj.nelement() / 1024**2}
                tensor_groups[key]['count'] += 1
        except:
            pass

    for (shape, dtype), info in sorted(tensor_groups.items(), key=lambda x: x[1]['count'] * x[1]['size_mb'], reverse=True):
        count = info['count']
        size_mb = info['size_mb']
        if count > 1:
            print(f"  {count}x{shape}, dtype={dtype}, size={count * size_mb:.2f} MB")
        else:
            print(f"  {shape}, dtype={dtype}, size={size_mb:.2f} MB")
            
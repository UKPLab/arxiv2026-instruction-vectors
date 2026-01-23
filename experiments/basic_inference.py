import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from arxiv2026_instruction_vectors import config, PROJECT_ROOT
from arxiv2026_instruction_vectors.data.load_datasets import InferenceDataset, load_task
from arxiv2026_instruction_vectors.metric_utils import (
    basic_accuracy,
    instruction_accuracy,
    judge_accuracy,
)


# Make deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# Load model
sampling_params = SamplingParams(
                        temperature=0, 
                        max_tokens=config.args.max_tokens,
                        logprobs=5,
                        )

model = LLM(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
print("Model:", model)

# Load task
task = load_task(config.args.task, config.args.subtask, tokenizer)

task_samples, task_instruction = (
    task.load_test(), 
    task.instruction_prompt
)
task_samples = InferenceDataset(task_samples)

# Prepare output paths for text responses and graphs
if "score_sheets" not in os.listdir(PROJECT_ROOT / "experiments/output/"):
    os.makedirs(PROJECT_ROOT / "experiments/output/score_sheets", exist_ok=True)
if config.args.task not in os.listdir(PROJECT_ROOT / "experiments/output/responses/"):
    os.makedirs(PROJECT_ROOT / "experiments/output/responses/" / config.args.task, exist_ok=True)
if config.args.task not in os.listdir(PROJECT_ROOT / "experiments/output/graphs/"):    
    os.makedirs(PROJECT_ROOT / "experiments/output/graphs/" / config.args.task, exist_ok=True)

# Make model subfolders
if config.short_name not in os.listdir(PROJECT_ROOT / "experiments/output/responses/" / config.args.task):
    os.makedirs(PROJECT_ROOT / "experiments/output/responses/" / config.args.task / config.short_name, exist_ok=True)
response_file = PROJECT_ROOT / "experiments/output/responses" / config.args.task / config.short_name / "response.txt"

score_sheet = PROJECT_ROOT / "experiments/output/score_sheets" / "{}.csv".format(config.short_name)
if config.short_name + ".csv" in os.listdir(PROJECT_ROOT / "experiments/output/score_sheets"):
    header = False
else:
    header = True

test_dataloader = DataLoader(task_samples, 
                             batch_size=config.args.test_batch_size,
                             shuffle=False,
                             )

write_to_file = {
    "model": config.short_name,
    "task": config.args.task if config.args.subtask is None else config.args.task + "_" + config.args.subtask,
    "num_samples": len(task_samples),
    }

batched_outputs = []
batched_targets = []
batched_options = []
batched_instructions = []

for batch in test_dataloader:
    inputs, targets, options, instructions = batch[0], batch[1], batch[2], batch[3]
    responses = model.generate(inputs, sampling_params=sampling_params)

    outputs = [response.outputs[0].text for response in responses]
    if config.args.task == "object_counting":
        options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        options = [str(o) for o in options]
    else:
        # vLLM duplicates each option * batch_size, need to correct this
        options = [opt[0] for opt in options]
    outputs = [response.outputs[0].text for response in responses]

    # Do basic postprocessing
    # Take first line only. Then strip spaces and remove periods
    postproc_outputs = []
    for o in outputs:
        o = o.split("\n")[0].strip()    # take first line
        o = re.sub(r"[\.\n,]", "", o)   # remove period and newline
        o = o.replace("<|im_end|>", "") # for olmo3
        o = re.sub(r"[\W]", "", o)  # remove any non-word chars
        o = o.replace("?| ", "")  # remove common gibberish pattern
        o = o.replace(".", "")
        if config.args.task in ["adjectives", "animals"]:
            o = o.lower()   # lowercase
        o = o.strip()
        postproc_outputs.append(o)
    outputs = postproc_outputs

    for idx in range(len(inputs)):
        output = responses[idx].outputs[0].text
        logprobs = responses[idx].outputs[0].logprobs
    
    batched_outputs.append(outputs)
    batched_targets.append(targets)
    batched_options.append(options)
    batched_instructions.append(instructions)

basic_acc = basic_accuracy(batched_outputs, batched_targets)
instr_acc = instruction_accuracy(batched_outputs, batched_options)
if config.args.subtask in ["adj_ant", "adj_comp"]:
    judge_acc = judge_accuracy(batched_outputs, batched_instructions, config.args.subtask)
elif config.args.task in ["metaphor_boolean", "implicatures", "snarks", "object_counting"]:
    judge_acc = judge_accuracy(batched_outputs, batched_instructions, config.args.task)
else:
    judge_acc = None

write_to_file["basic_accuracy"] = basic_acc
write_to_file["instruction_accuracy"] = instr_acc
write_to_file["judge_accuracy"] = judge_acc

final_df = pd.DataFrame(write_to_file, columns=write_to_file.keys(), index=[0])
final_df.to_csv(score_sheet, sep=",", mode='a', header=header, index=False)

print("OVERALL BA:", basic_acc)
print("OVERALL INSTR ACC:", instr_acc)
print("OVERALL JUDGE ACC:", judge_acc)


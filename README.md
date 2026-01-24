# arxiv2026_instruction_vectors
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/arxiv2026-instruction-vectors)](https://github.com/UKPLab/arxiv2026-instruction-vectors/blob/main/LICENSE)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This is the accompanying code repository for the paper [Patches of Nonlinearity: Instruction Vectors in Large Language Models](https://github.com/UKPLab/arxiv2026-instruction-vectors).

> **Abstract:** Despite the recent success of instruction-tuned language models and their ubiquitous usage, very little is known of how models process instructions internally. In this work, we address this gap from a mechanistic point of view by investigating how instruction-specific representations are constructed and utilized in different stages of post-training: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Via causal mediation, we identify that instruction representation is fairly localized in models. These representations, which we call _Instruction Vectors_ (IVs), demonstrate a curious juxtaposition of linear separability along with non-linear causal interaction, broadly questioning the scope of the linear representation hypothesis commonplace in mechanistic interpretability. To disentangle the non-linear causal interaction, we propose a novel method to localize information processing in language models that is free from the implicit linear assumptions of patching-based techniques. We find that, conditioned on the task representations formed in the early layers, different information pathways are selected in the later layers to solve that task, i.e., IVs act as _circuit selectors_.

Contact person: [Irina Bigoulaeva](mailto:irina.bigoulaeva@gmail.com) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started

Install the necessary dependencies in `uv.lock`:

```bash
uv sync
```

#### Optional vLLM
A subset of our experiments (i.e. `experiments/basic_inference.py`) requires [vLLM](https://docs.vllm.ai/en/latest/) as a dependency. It's recommended to install this in a clean environment first, as specific CUDA versions are required that may introduce incompatibilities. For this reason, our `uv.lock` file excludes vLLM.

We recommend installing vLLM in a separate environment for running the inference experiments.

```bash
uv pip install vllm
```


## Usage

The paper's experiments and graphs can be reproduced by running the scripts under the `experiments` folder. The outputs of the scripts are saved under `experiments/output`.

From the project root directory:
```bash
source .venv/bin/activate
cd experiments
```

Now, the scripts can be run from the command line.

## Running experiments

Before running experiments, specify necessary hyperparameters in `args.py`.

`--model_idx`: Specify the index of the model (as listed in `config.py`).

`--task`: Name of the main task (corresponds to dataset/task in data/local_tasks/).
  - `adjectives`
  - `animals`
  - `metaphor_boolean`
  - `object_counting`
  - `implicatures`
  - `snarks`

`--subtask`: Name of the subtask if the main task is either `adjectives` or `animals`.
  - `adjectives`: `adj_comp` or `adj_ant`
  - `animals`: `anim_color` or `can_fly`

`--num_samples`: Number of task samples to use in the experiment.

Additionally, there are some parameters that are required for specific experiments.

### Activation Patching

`--num_choices`: When conducting activation patching, define whether to do 2-layer patching (num_choices=2) or 3-layer patching (num_choices=3). This represents a combinatorial n-choose-k search of tuple pairs/triplets among the *n* model layers. 

```bash
# 2-layer patching on Olmo-2 1B, on 100 samples from the Adjective: Comparative task
uv run activation_patching.py --model_idx 17 --task "adjectives" --subtask "adj_comp" --num_choices 2 --num_samples 100
```

```bash
# 3-layer patching on Olmo-2 7B DPO, on 100 samples from the Metaphor Boolean task
uv run activation_patching.py --model_idx 19 --task "metaphor_boolean" --num_choices 3 --num_samples 100
```

### Path Tracing

`--start_pos`: The token position from which to start tracing. Due to long runtimes and an exponentially larger number of paths towards the beginning of the prompt, we recommend tracing from later token positions.

`--end_pos`: The token position at which to stop tracing (inclusive).

`--threshold_rank`: The rank threshold for saving paths. Default is 100, but can be set to lower. Setting to higher will result in more paths being computed and greater runtimes/memory load.

The code always assumes that we are tracing one data sample at a time. Therefore, iteration over many samples is done within the bash script.

Tracing is only done on contrastive tasks, and both subtasks are done by default. So, no `--subtask` parameter is passed.

```bash
# Do path tracing with Olmo-2 1B DPO on the Adjectives tasks, on samples 0 - 49 (inclusive).

for i in $(seq 0 49);
do
        uv run path_tracing.py --model_idx 16 --task "adjectives" --tracing_sample_idx $i --start_pos 11 --end_pos -1 --threshold_rank 100
done

```


### Linear Probe and Dimensionality Reduction Experiments

For our linear probe experiments, we must produce and load from a dataset of varied instructional samples. These variations of the instruction can either keep the label space constant (e.g. keep a yes/no question a yes/no question), or can change the label space (e.g. turn a yes/no question into an T/F question). 

`--varying`: 
  * `instructions` - if label space is not changed
  * `labels` - if label space is changed.

Note that in our paper, we only vary instructions, so there is a minimal selection of tasks for which a varied label space is predefined in the codebase.

For our geometric analysis experiments, we must specify which dimensionality reduction method to use.

`--reduction_method`: 
  * `pca`
  * `tsne`
  * `lda`

Example command:

```bash
python -m vector_space_analysis.py --reduction_method "lda"
```

## Cite

If you found our work helpful, please cite our paper:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

"""
Logistic regression based on the instruction representations.
Goal is to determine whether the categories are linearly separable.

"""

import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, disable_caching, load_from_disk
from nnsight import LanguageModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from arxiv2026_instruction_vectors.data.load_datasets import config, load_task

disable_caching()
   

def make_train_test(tasks, save_name, varying, model_component, layer=None):
    """
    Make the train and test sets for the probe.
    :param layer: Optionally train on hidden states from a specific layer onlys
    """

    if layer is not None:
        layers = [layer]
        save_path = f"hidden_state_datasets/{config.short_name}_{model_component}_{varying}_lyr_{layers[0]}/"
    else:
        layers = [l for l in range(config.n_layers)]
        save_path = f"hidden_state_datasets/{config.short_name}_{model_component}_{varying}_{save_name}/"

    os.makedirs(save_path, exist_ok=True)
    print("Made output dir:", save_path)

    data = {
        "vector": [],
        "cluster": [],
        "layer": []
    }

    # Load model
    model = LanguageModel(config.model_name, device_map="auto")
    print("Model:", model)

    for (task_name, subtask) in tasks:
        print("Doing task:", task_name, subtask)
        # Load task instructions
        task = load_task(task_name, subtask)

        if varying == "labels":
            task_instructions = task.varied_labels
        elif varying == "instructions":
            task_instructions = task.varied_instructions

        print("Varied instructions:", task_instructions)

        # For each instruction, get states over all layers and plot PCA
        for task_instruction in tqdm(task_instructions):
            for targ_layer_idx in layers:
                with model.trace(task_instruction) as tracer:
                    if model_component == "resid_post":
                        # Take two [0] to remove the unnecessary 1st dimension for PCA. From torch.Size([1, 10, 896]) -> torch.Size([10 (len instr tokens), 896 (hidden dim)])
                        # Then take [-1] to get the final token of the instruction
                        instr_hs = model.model.layers[targ_layer_idx].output[0][-1].cpu().save()     # output shape (hidden_state_info,)
                    elif model_component == "attn_out":
                        instr_hs = model.model.layers[targ_layer_idx].self_attn.output[0][0][-1].cpu().save()
                    elif model_component == "mlp":
                        instr_hs = model.model.layers[targ_layer_idx].mlp.output[0][0][-1].cpu().save()

                data["vector"].append(instr_hs.cpu().detach())
                if subtask is not None:
                    data["cluster"].append(subtask)
                else:
                    data["cluster"].append(task_name)
                data["layer"].append(targ_layer_idx)
                del instr_hs
                gc.collect()
                torch.cuda.empty_cache()
        print("Finished task:", task_name)
    
    dataset = Dataset.from_dict(data)
    dataset.train_test_split(test_size=0.2)
    dataset.save_to_disk(save_path)
    print("Saved dataset:", dataset, len(dataset))


def train_and_test_probe(model_name, varying, model_component, num_epochs, layer=None, learning_rate=0.001):

    if layer is not None:
        lyr_suff = f"_lyr_{layer}"
    else:
        lyr_suff = ""

    data_path = f"hidden_state_datasets/{model_name}_{model_component}_{varying}{lyr_suff}/"
    scores_path = f"hidden_state_datasets/scores_{model_name}_{model_component}_{varying}.csv"

    scores_df = {
        "model": [],
        "accuracy": [],
        "num_epochs": [],
        "learn_rate": [],
        "num_train": [],
        "num_test": [],
        "layer": [],
    }

    dataset = load_from_disk(data_path)
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=0.2)

    train = dataset["train"]
    test = dataset["test"]

    def _label_to_idx(label):
        l2i = {"adj_ant": 0,
               "adj_comp": 1,
               "anim_color": 2,
               "can_fly": 3}
        return l2i[label]

    train_data = torch.tensor([np.array(vec) for vec in train["vector"]], dtype=torch.float32)
    train_labels = torch.tensor([_label_to_idx(c) for c in train["cluster"]], dtype=torch.long)

    test_data = torch.tensor([np.array(vec) for vec in test["vector"]], dtype=torch.float32)
    test_labels = torch.tensor([_label_to_idx(c) for c in test["cluster"]], dtype=torch.long)

    probe = LogisticRegression()
    probe.fit(train_data, train_labels)
    score = probe.score(test_data, test_labels)

    print("SCORE:", score)

    scores_df["model"].append(model_name)
    scores_df["accuracy"].append(score)
    scores_df["num_epochs"].append(num_epochs)
    scores_df["learn_rate"].append(learning_rate)
    scores_df["num_train"].append(train_data.shape[0])
    scores_df["num_test"].append(test_data.shape[0])
    scores_df["layer"].append(lyr_suff.replace("_lyr_", ""))

    scores_df = pd.DataFrame(scores_df)
    scores_df.to_csv(scores_path, mode="a", header=True)
    
    return score

def lda_analysis(models, varying, model_component, model_size, layer=None):

    if layer is not None:
        lyr_suff = f"_lyr_{layer}"
    else:
        lyr_suff = ""

    # Matplotlib figure
    fig, axs = plt.subplots(1, len(models), figsize=(20, 10))
    plt.subplots_adjust(left=0.2, top=0.9, #hspace=0.4
                        )

    # Prepare colors for legend
    color_per_task = {
        "metaphor_boolean": "#FF4882",
        "implicatures": "#C3FF53",
        "object_counting": "#ADF2E6",
        "snarks": "#FFDC9F",
        "adj_comp": "#C72FB5",
        "adj_ant": "#FDD0FF",
        "anim_color": "#968CFF",  
        "can_fly":  "#2F00CA",
        "m_color": "#00F2B9",
        "m_edible": "#00A26F",
        "math_addition": "#00D9FF",
        "math_purple": "#0E88E5",
    }
    for idx in range(len(axs)):

        # Load saved data for model
        model_name = models[idx]

        # Load contrastive tasks
        data_path = f"hidden_state_datasets/{model_name}_{model_component}_{varying}{lyr_suff}/"
        contrastive_dataset = load_from_disk(data_path)

        # Load bb tasks
        bb_data_path = f"hidden_state_datasets/{model_name}_{model_component}_{varying}_bb/"
        bb_dataset = load_from_disk(bb_data_path)
        bb_dataset = bb_dataset.remove_columns("layer")

        dataset = concatenate_datasets([contrastive_dataset, bb_dataset])

        concat_data = np.array(dataset["vector"])
        labels = dataset["cluster"]
        num_classes = len(set(labels))

        # Train and fit LDA classifier
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(concat_data, labels)
        hs_reduced = clf.transform(concat_data)

        explained_variance = clf.explained_variance_ratio_
        print("Explained variance:", explained_variance)
        variance_out_path = f"src/graphs/{model_name}_{model_component}_varied_{varying}_lda_variance.pt"
        torch.save(explained_variance, variance_out_path)

        print("Saved explained variance to:", variance_out_path)

        # Plot the reduced representations
        for i in range(len(hs_reduced)):
            axs[idx].scatter(
                hs_reduced[i, 0],
                hs_reduced[i, 1],
                c=color_per_task[dataset[i]["cluster"]],
                #cmap=cmap_per_task[colormap_id],
                label=dataset[i]["cluster"]
                )
            axs[idx].set_title(model_name)

    # Get legend labels
    handles, labels = axs[-1].get_legend_handles_labels()
    unique_idxs = list(range(0, hs_reduced.shape[0], hs_reduced.shape[0]//num_classes))
    unique_handles = [handles[unique_idxs[i]] for i in range(len(unique_idxs))]
    unique_labels = [labels[unique_idxs[i]] for i in range(len(unique_idxs))]

    fig.legend(unique_handles, unique_labels, loc='outside lower center', ncols=4)

    out_path = f"src/graphs/{model_size}_{model_component}_varied_{varying}_lda.pdf"

    fig.savefig(out_path, dpi=200)
    print(f"Saved fig to: {out_path}")


if __name__ == "__main__":

    if config.args.make_dataset:
        make_train_test([
                        ("metaphor_boolean", None),
                        ("implicatures", None),
                        ("object_counting", None),
                        ("snarks", None),
                        ],
                        save_name="bb",
                        varying="instructions",
                        model_component=config.args.model_component,
                        )
        
    elif config.args.do_probe:
        #for layer in [0, 1, 2, 4, 7, 15]:
        for model in ["olmo-7b", "olmo-7b-sft", "olmo-7b-dpo"]:
            train_and_test_probe(
                model_name=model, #config.short_name,
                varying="instructions", 
                model_component=config.args.model_component,
                num_epochs=1,
                #layer=0
            )

    else:
        # Do LDA
        models = ["olmo-7b", "olmo-7b-sft", "olmo-7b-dpo"]
        lda_analysis(
            models=models,
            varying="instructions", 
            model_component="resid_post",
            model_size="7b",
            layer=None,  
        )


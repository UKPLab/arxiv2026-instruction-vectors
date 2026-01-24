"""
Passively get vector representations of single layers given rephrasals of the instruction.
No activation patching is done.
Plot using PCA, TSNE, etc. 
"""
import gc
from collections import defaultdict

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from nnsight import CONFIG, LanguageModel, util
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from tqdm import tqdm

from src import config
from src.data.load_datasets import load_task

def get_instr_states(tasks, method, varying, layerwise_graph, model_component):
    """
    We can graph layerwise across all tasks (all tasks per layer),
    or we can graph all layers + all tasks as one
    """
    # Load model
    model = LanguageModel(config.model_name, device_map="auto")

    # Prepare scatter figure
    if layerwise_graph:
        n_rows = config.n_layers // 4

        # Matplotlib figure
        fig, axs = plt.subplots(
            n_rows, 4, figsize=(15, 15), 
            #subplot_kw={"projection": "3d"}
            )
        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=None, hspace=0.4)

        # Plotly figure
        plotly_fig = make_subplots(
            rows=n_rows, cols=4,
            subplot_titles=tuple([f"Layer {idx}" for idx in range(config.n_layers)])
            )

    else:
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 12), #subplot_kw={"projection": "3d"}
            )

    # Prepare color mapping - Separate tasks by hues
    # Early vs. late layers by gradients within the hue
    color_per_layer = [i for i in range(config.n_layers)]
    cmap_per_task = {
        "antonyms": "Purples",
        "metaphor_boolean": "Blues",
        "implicatures": "Greens",
        "object_counting" : "Reds", 
    }
    tasks_to_colors = [i for i in range(len(tasks))]
    
    # Prepare colors for legend
    legend_colors = {}
    color_per_task = {
        "antonyms": "#FF2970",
        "metaphor_boolean": "#594AFF",
        "implicatures": "#3DFF6A",  #"#FDD03C"
        "object_counting": "#3FF9FF",
        "adj_comp": "#E60000",
        "adj_ant": "#FFA3A3",
        "anim_color": "#0728E2",
        "can_fly": "#B4B8FF",
    }

    reduced_hs_all_tasks = defaultdict(list)
    df_hs_all_tasks = {
        "x_coord": [],
        "y_coord": [],
        #"z_coord": [],
        "task": [],
        "layer": [],
        "instruction": [],
    }

    concat_data_per_task = {}

    for (task_name, subtask) in tasks:
        # Load task instructions
        task = load_task(task_name, subtask)
        colormap_id = subtask if subtask is not None else task_name

        if varying == "labels":
            task_instructions = task.varied_labels
        elif varying == "instructions":
            task_instructions = task.varied_instructions

        instruction_states = defaultdict(list)
    
        # For each instruction, get states over all layers and plot PCA
        for task_instruction in tqdm(task_instructions):
            prompt_tokens = model.tokenizer.encode(task_instruction)

            for targ_layer_idx in range(config.n_layers):
                with model.trace(prompt_tokens) as tracer:
                    if model_component == "resid_post":
                        # Take two [0] to remove the unnecessary 1st dimension for PCA. From torch.Size([1, len(prompt), hid_dim]) -> torch.Size([len(prompt), hid_dim])
                        # Then take [-1] to get the final token of the instruction
                        instr_hs = model.model.layers[targ_layer_idx].output[0][0][-1].save()     # output shape (hidden_state_info,)
                    elif model_component == "attn_out":
                        instr_hs = model.model.layers[targ_layer_idx].self_attn.output[0][0][-1].cpu().save()
                    elif model_component == "mlp":
                        instr_hs = model.model.layers[targ_layer_idx].mlp.output[0][0][-1].cpu().save()
                instruction_states[task_instruction].append(instr_hs.cpu().detach())
                del instr_hs
                gc.collect()
                torch.cuda.empty_cache()
            
            # Compute PCA for the instruction
            concat_data = np.array(instruction_states[task_instruction]) # shape (24, 8960)

            concat_data_per_task[subtask] = concat_data

            if method == "pca":
                pca = PCA(n_components=2)
                hs_reduced = pca.fit_transform(concat_data)
            elif method == "tsne":
                hs_reduced = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(concat_data)
            elif method == "lda":
                clf = LinearDiscriminantAnalysis(solver="svd")
                hs_reduced = clf.fit_transform(concat_data)

            #print("HS_REDUCED:", hs_reduced)
            reduced_hs_all_tasks[task_name].append(hs_reduced)

            for layer_idx in range(len(hs_reduced)):
                x = hs_reduced[layer_idx][0]
                y = hs_reduced[layer_idx][1]
                #z = hs_reduced[layer_idx][2]

                df_hs_all_tasks["layer"].append(layer_idx)
                df_hs_all_tasks["x_coord"].append(x)
                df_hs_all_tasks["y_coord"].append(y)
                #df_hs_all_tasks["z_coord"].append(z)
                df_hs_all_tasks["task"].append(task_name)
                df_hs_all_tasks["instruction"].append(task_instruction)

            if layerwise_graph:
                # Plot each layer of hs_reduced on its own axis
                layer_idx = -1
                for row_idx in range(n_rows):
                    for col_idx in range(axs.shape[1]):

                        # Matplotlib figure
                        layer_idx += 1
                        scatter = axs[row_idx][col_idx].scatter(
                        hs_reduced[layer_idx, 0],
                        hs_reduced[layer_idx, 1],
                        #hs_reduced[layer_idx, 2],
                        c=color_per_task[colormap_id],
                        #cmap=cmap_per_task[task_name],                  
                        )
                        axs[row_idx][col_idx].set(title=f"Layer {layer_idx}")

            else:
                # Plot all of hs_reduced on the same axis
                # Matplotlib
                scatter = ax.scatter(
                    hs_reduced[:, 0],
                    hs_reduced[:, 1],
                    #hs_reduced[:, 2],
                    c=color_per_layer,
                    cmap=cmap_per_task[colormap_id],
                    )

        # Make graph legend
        if layerwise_graph:
            pass
        else:
            # Define colors for the graph
            cmap = mpl.colormaps[cmap_per_task[colormap_id]]
            last_color = cmap(0.6)
            legend_colors[colormap_id] = last_color
            tasks_to_colors.append(mpatches.Circle((1,1), color=legend_colors[colormap_id], label=task_name))
            fig.legend(handles=tasks_to_colors,
                    loc="outside lower center",
                    ncols=4
                    )
        
        print("Finished task:", task_name)

    # Finished all tasks - now do plotly figure
    # Plotly prefers a dataframe
    plotly_fig = px.scatter(
        df_hs_all_tasks, 
        x="x_coord", 
        y="y_coord", 
        #z="z_coord", 
        color="task", 
        facet_col="layer", 
        facet_col_wrap=4,
        hover_data="instruction",
        facet_row_spacing=0.02,
        facet_col_spacing=0.02,
        )
 
    # Set title

    # Matplotlib
    figtitle = f"Varied {varying.capitalize()} - {model_component.capitalize()} - {config.short_name.capitalize()} - {method.upper()}"
    fig.suptitle(figtitle, y=0.95)

    # Plotly
    plotly_fig.update_layout(height=2000, width=1700,
                    title_text=figtitle)
    

    # Define out path and save
    out_path = f"output/graphs/{config.short_name}_{model_component}_varied_{varying}_{method}"
    fig.savefig(out_path, dpi=200)

    plotly_fig.show()
    plotly_fig.write_html(out_path + ".html")
    return

get_instr_states([
                  ("adjectives", "adj_comp"),
                  ("adjectives", "adj_ant"),
                  ("animals", "anim_color"),
                  ("animals", "can_fly"),
                  ],
                  method=config.args.reduction_method,
                  varying="instructions",
                  layerwise_graph=True,
                  model_component=config.args.model_component
                  )
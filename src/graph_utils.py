"""
Functions for creatiing various graphs used in our analyses.
"""

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def graph_inference_scores(scores_df, instr_df, save_path, model_groups):
    tasks = list(scores_df.index)
    n_tasks = len(tasks)
    n_rows = len(model_groups)

    task_colors = [
        "#45BC96", "#68EDB3", "#0055FF", "#9EFAFA",
        "#2F00CA", "#968CFF", "#C72FB5", "#FDD0FF",
    ]
    colors = {task: color for task, color in zip(tasks, task_colors)}

    width = 0.8 / n_tasks
    center_offset = width * n_tasks / 2

    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), layout="constrained")
    if n_rows == 1:
        axs = [axs]

    for ax, chunk in zip(axs, model_groups):
        x = np.arange(len(chunk))

        for task_idx, task in enumerate(tasks):
            offset = width * task_idx
            instr_vals = [float(instr_df.loc[task, m]) for m in chunk]
            basic_vals = [float(scores_df.loc[task, m]) for m in chunk]
            ax.bar(x + offset, instr_vals, width, color="gainsboro", edgecolor="black", linewidth=0.3)
            ax.bar(x + offset, basic_vals, width, label=task, color=colors[task], edgecolor="black", linewidth=0.3)

        shortened = [m.replace("instruct", "it") for m in chunk]
        ax.axhline(y=50, color="gray", linestyle="dashed")
        ax.set_xticks(x + center_offset, shortened)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.yaxis.minorticks_on()

    # Grab only the colored (basic accuracy) bar handles — one per task
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[-n_tasks:], labels[-n_tasks:], loc="outside lower center", ncols=2)
    plt.savefig(save_path, dpi=200)
    return


def make_mlp_heatmap(model_scores,
                     save_path,
                     num_layers,
                     num_tasks,
                     task_names,
                     model_names,
                     setting,
                     transpose_axis,
                     colormap='viridis',
                     ):
    def _make_value_matrix(model_scores):
        values_matrix = []
        for row_head in range(num_layers):
            row = [model_scores[row_head, i] for i in range(num_layers)]
            values_matrix.append(row)
        return values_matrix

    values_matrices = [
        [_make_value_matrix(scores_model[task_idx]) for task_idx in range(len(scores_model))]   # shape (1, n_tasks)
        for scores_model in model_scores
    ] # shape (n_models, n_tasks)

    for scores_model in model_scores:
        for task_idx in range(len(scores_model)):
            scores = scores_model[task_idx]
            scores = [item for item in scores_model[task_idx].items() if not math.isnan(item[1])]
            scores.sort(key=lambda item: item[1], reverse=True)

    values_matrices = np.array(values_matrices)  # shape (num_models, num_tasks, n_layers, n_layers)

    if transpose_axis:
        # Tasks are rows, models are columns
        fig, axs = plt.subplots(num_tasks, values_matrices.shape[0], 
                                figsize=(10, 10),
                                )
        values_matrices = np.transpose(values_matrices, axes=(1, 0, 2, 3))  # shape (num_tasks, num_models, n_layers, n_layers)

        # Name the x and y axes accordingly
        for idx in range(len(model_names)):
            axs[0][idx].set_title(model_names[idx])
        for idx in range(len(task_names)):
            axs[idx][0].set_ylabel(task_names[idx], size='large')

    else:
        # Models are rows, tasks are columns
        fig, axs = plt.subplots(values_matrices.shape[0], num_tasks, 
                                figsize=(15, 15),
                                )
        # Name the x and y axes accordingly
        for idx in range(len(task_names)):
            axs[0][idx].set_title(task_names[idx])
        for idx in range(len(model_names)):
            axs[idx][0].set_ylabel(model_names[idx], size='large')
    
    cmap = colormap

    # Make a normalized colorbar
    flattened = np.array(values_matrices).flatten()
    max_val = np.nanmax(flattened)
    min_val = np.nanmin(flattened)  # we set the min to 0 for clearer plots
    norm = mpl.colors.Normalize(vmin=0, vmax=max_val)

    images = []
    for row_idx in range(values_matrices.shape[0]):
        for col_idx in range(values_matrices.shape[1]):
            img = axs[row_idx, col_idx].imshow(
                values_matrices[row_idx, col_idx], 
                cmap=cmap, 
                #norm=norm,
            )
            images.append(img)
            fig.colorbar(
                img, 
                ax=axs[row_idx, col_idx], 
                location='right', 
                fraction=0.046, 
                pad=0.04, 
                #norm=norm,
            )

    # Plot the data on a 16x16 (or 32x32 for 7B) grid
    for ax in axs.flat:
        ax.set_xticks(range(num_layers))
        if num_layers == 16:
            labels = [str(i) if i % 2 == 0 else "" for i in range(num_layers)]
        elif num_layers == 32:
            labels = [str(i) if i % 4 == 0 else "" for i in range(num_layers)]
        ax.set_xticklabels(labels=labels)
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(labels=labels)
    
    fig.tight_layout()
    if len(model_scores) == 3:
        fig.subplots_adjust(hspace=0.1, wspace=0.4, bottom=0.4)
    elif len(model_scores) == 2:
        fig.subplots_adjust(hspace=0.1, wspace=0.4, bottom=0.6)
    plt.savefig(save_path, dpi=200)
    return

def head_change_heatmap(head_changes_all, num_samples, decoded_tokens, model, task, sample_idx, num_heads, num_layers, save_path):
    # Set fig properties 
    print(list(head_changes_all.keys()))
    n_cols = 2
    if len(head_changes_all) % 2 == 0:
        n_rows = len(head_changes_all) // 2
        fig, axs = plt.subplots(
            n_rows, n_cols, 
            figsize=(16, 16),
        )
        dropped_last = False
    else:
        n_rows = int(np.ceil(len(head_changes_all) / 2))
        fig, axs = plt.subplots(
            n_rows, n_cols, 
            figsize=(16, 16),
        )
        fig.delaxes(axs[-1,1])
        dropped_last = True

    # Get data
    layers = [l for l in range(num_layers)]
    heads = list(reversed([h for h in range(num_heads)]))

    # Collect the data matrix for each token
    values_matrices = []
    max_all_tokens = []

    for src_token in head_changes_all:
        values_matrix = [[0 for i in range(num_heads)] for layer in layers]
        for layer in layers:
            if layer in head_changes_all[src_token]:#head_changes_s:
                for head_idx in range(len(head_changes_all[src_token][layer])):
                    values_matrix[layer][head_idx] += head_changes_all[src_token][layer][head_idx]

        # Reverse the y-axis to make graph more readable
        values_matrix = [list(reversed(l)) for l in values_matrix]
        
        max_val = max(max(values_matrix))
        values_matrices.append(values_matrix)
        max_all_tokens.append(max_val)

    # Make a colorbar
    cmap = mpl.colormaps['GnBu']
    flattened = np.array(values_matrices).flatten()
    max_n_changes = np.max(flattened)

    # Make it discrete normalized only if doing 1 sample (not avg)
    if max_n_changes > 1:
        bounds = np.arange(0, max_n_changes + 1)
    else:
        bounds = [0,1]
    norm = mpl.colors.Normalize(vmin=0, vmax=max_n_changes)
    
    # Make the plots
    images = []
    token_idx = -1
    for row_idx in range(n_rows):
        for col_idx in range(axs.shape[1]):
            token_idx += 1
            if token_idx == len(head_changes_all) and dropped_last:
                break
            else:
                #print("ROW IDX:", row_idx, "COL IDX:", col_idx)
                token = list(head_changes_all.keys())[token_idx]
                decoded_token = decoded_tokens[token]
                images.append(axs[row_idx][col_idx].imshow(values_matrices[token_idx], cmap=cmap, norm=norm))
                axs[row_idx][col_idx].set_yticks(range(num_layers), labels=layers)
                axs[row_idx][col_idx].set_xticks(range(num_heads), labels=heads)
                axs[row_idx][col_idx].set(title=f"'{decoded_token}' ({token})")  # formerly "token"
                axs[row_idx][col_idx].set_xlabel("Heads")
                axs[row_idx][col_idx].set_ylabel("Layers")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0)
    fig.colorbar(images[1], ax=axs, location='right', fraction=.1)
    fig.suptitle(f"# Active Attn Head Changes Across Paths for {num_samples} Samples\n{model.capitalize()} - {task.capitalize()}", va='top')
    plt.savefig(save_path, dpi=200)
    print("Fig saved to:", save_path)
    plt.show()
    return

# For a single token, track the head changes across two contrastive tasks
def single_query_heatmap(head_changes_all, query, decoded_tokens, model, task1, task2, num_heads, num_layers, save_path):

    # Set fig properties
    print(list(head_changes_all.keys()))
    n_cols = 2
    if len(head_changes_all) % 2 == 0:
        n_rows = len(head_changes_all) // 2
        fig, axs = plt.subplots(
            n_rows, n_cols, 
            figsize=(16, 16),
            #layout="constrained"
        )
        dropped_last = False
    else:
        n_rows = int(np.ceil(len(head_changes_all) / 2))
        fig, axs = plt.subplots(
            n_rows, n_cols, 
            figsize=(16, 16),
            #layout="constrained"
        )
        fig.delaxes(axs[-1,1])
        dropped_last = True

    # Get data
    layers = [l for l in range(num_layers)]
    heads = list(reversed([h for h in range(num_heads)]))

    # Collect the data matrix for each token
    values_matrices = []
    max_all_tokens = []

    for src_token in head_changes_all:
        values_matrix = [[0 for i in range(num_heads)] for layer in layers]
        for layer in layers:
            if layer in head_changes_all[src_token]:#head_changes_s:
                for head_idx in range(len(head_changes_all[src_token][layer])):
                    values_matrix[layer][head_idx] += head_changes_all[src_token][layer][head_idx]

        # Reverse the y-axis to make graph more readable
        values_matrix = [list(reversed(l)) for l in values_matrix]
        
        max_val = max(max(values_matrix))
        values_matrices.append(values_matrix)
        max_all_tokens.append(max_val)

    # Make a colorbar
    cmap = mpl.colormaps['GnBu']
    flattened = np.array(values_matrices).flatten()
    max_n_changes = np.max(flattened)

    # Make it discrete normalized only if doing 1 sample (not avg)
    if max_n_changes > 1:
        bounds = np.arange(0, max_n_changes + 1)
    else:
        bounds = [0,1]
    norm = mpl.colors.Normalize(vmin=0, vmax=max_n_changes)
    
    # Make the plots
    images = []
    token_idx = -1
    for row_idx in range(n_rows):
        for col_idx in range(axs.shape[1]):
            token_idx += 1
            if token_idx == len(head_changes_all) and dropped_last:
                break
            else:
                #print("ROW IDX:", row_idx, "COL IDX:", col_idx)
                token = list(head_changes_all.keys())[token_idx]
                decoded_token = decoded_tokens[token]
                images.append(axs[row_idx][col_idx].imshow(values_matrices[token_idx], cmap=cmap, norm=norm))
                axs[row_idx][col_idx].set_yticks(range(num_layers), labels=layers)
                axs[row_idx][col_idx].set_xticks(range(num_heads), labels=heads)
                axs[row_idx][col_idx].set(title=f"'{decoded_token}' ({token})")  # formerly "token"
                axs[row_idx][col_idx].set_xlabel("Heads")
                axs[row_idx][col_idx].set_ylabel("Layers")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0)
    fig.colorbar(images[1], ax=axs, location='right', fraction=.1)
    fig.suptitle(f"# Active Attn Head Changes Across Paths in {model.capitalize()}\n'{query}' - {task1.capitalize()}/{task2.capitalize()}", va='top')
    plt.savefig(save_path, dpi=200)
    print("Fig saved to:", save_path)
    plt.show()
    return

def active_heads_heatmap_tinst_only(
        heads_info, 
        task_pair, 
        longest_prompt,
        prompt_1, 
        prompt_2, 
        num_heads, 
        num_layers, 
        save_path
        ):
    
    # Set fig properties
    n_cols = 2
    n_rows = 1

    fig, axs = plt.subplots(
        n_rows, n_cols, 
        figsize=(60, 60),
        #layout="constrained"
    )
    # Get data
    layers = [l for l in range(num_layers)]
    heads = list(reversed([h for h in range(num_heads)]))
    prompt_pair = [prompt_1, prompt_2]

    both_tasks_per_token = {}
    for src_token in heads_info:  # The files that exist. The ones that don't have value 0
        both_tasks_this_token = []
        for task in task_pair:
            try:
                task_subdict = heads_info[src_token][task]
                task_info_this_token = [[task_subdict[src_tok][head] for head in task_subdict[src_tok]] for src_tok in task_subdict]
                
            except KeyError:
                # This src_token pos has no high-ranking paths for the task
                task_info_this_token = [[[0.0, 0.0] for h in range(num_heads)] for src_tok in range(longest_prompt)]
            both_tasks_this_token.append(task_info_this_token)
        both_tasks_per_token[src_token] = both_tasks_this_token

    # Make a colorbar
    cmap = mpl.colormaps['RdPu']
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
    
    # Make the plots
    images = []
    t_inst_tokens = [[8, 8], [10, 11]]
    for row_idx in range(n_rows):
        col_shape = t_inst_tokens[row_idx]

        print("Both tasks per token:", list(both_tasks_per_token.keys()))
        for col_idx in range(n_cols):
            # Select a token for the row,col
            token_idx = col_shape[col_idx]

            prompt = prompt_pair[col_idx]
            try:
                decoded_token = prompt["decoded"].iloc[token_idx]
                data_one_task = np.array(both_tasks_per_token[token_idx][col_idx])


                # Reverse the presentation order of heads to make the heatmap axes more readable
                # This doesn't change the data itself
                # But must be paired with "heads = list(reversed([h for h in range(num_heads)]))" or the graph will be wrong
                data_one_task = np.flip(data_one_task, axis=1)
            except IndexError:
                print(f"No token at pos {token_idx} for prompt {prompt} of len {len(prompt)}. Setting to zeros.")
                data_one_task = np.zeros(shape=(num_heads, num_heads))

            im = axs[col_idx].imshow(data_one_task, cmap=cmap, norm=norm)
            images.append(im)
            fig.colorbar(im, ax=axs[col_idx], location='right', fraction=0.046, pad=0.04, norm=norm)
            axs[col_idx].set_yticks(range(num_layers), labels=layers)
            axs[col_idx].set_xticks(range(num_heads), labels=heads)
            axs[col_idx].set(title=f"Tok {token_idx} '{decoded_token}'")  # formerly "token"
            axs[col_idx].set_xlabel("Heads")
            axs[col_idx].set_ylabel("Layers")

            # Ad-hoc label for colorbar
            axs[col_idx].text(19.3,7,"% activity", size=10, verticalalignment='center', rotation=270)
            

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.3, wspace=0.4, right=0.2)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Fig saved to:", save_path)
    return


def active_heads_heatmap(
        heads_info, 
        query, 
        task_pair, 
        longest_prompt,
        prompt_1, 
        prompt_2, 
        model,
        sample_idx,
        num_heads, 
        num_layers, 
        save_path
        ):
    
    # Set fig properties
    n_cols = 2
    n_rows = len(heads_info)

    fig, axs = plt.subplots(
        n_rows, n_cols, 
        figsize=(60, 60),
        #layout="constrained"
    )
    # Get data
    layers = [l for l in range(num_layers)]
    heads = list(reversed([h for h in range(num_heads)]))
    prompt_pair = [prompt_1, prompt_2]

    both_tasks_per_token = {}
    for src_token in heads_info:  # The files that exist. The ones that don't have value 0
        both_tasks_this_token = []
        for task in task_pair:
            try:
                task_subdict = heads_info[src_token][task]
                task_info_this_token = [[task_subdict[src_tok][head] for head in task_subdict[src_tok]] for src_tok in task_subdict]
                
            except KeyError:
                # This src_token pos has no high-ranking paths for the task
                task_info_this_token = [[[0.0, 0.0] for h in range(num_heads)] for src_tok in range(longest_prompt)]
            both_tasks_this_token.append(task_info_this_token)
        both_tasks_per_token[src_token] = both_tasks_this_token

    # Make a colorbar
    cmap = mpl.colormaps['RdPu']
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
    
    # Make the plots
    images = []
    for row_idx in range(len(both_tasks_per_token)):
        # Select a token for this row
        token_idx = list(both_tasks_per_token.keys())[row_idx]

        print("Both tasks per token:", list(both_tasks_per_token.keys()))
        for col_idx in range(axs.shape[1]):
            prompt = prompt_pair[col_idx]
            try:
                decoded_token = prompt["decoded"].iloc[token_idx]
                data_one_task = np.array(both_tasks_per_token[token_idx][col_idx])


                # Reverse the presentation order of heads to make the heatmap axes more readable
                # This doesn't change the data itself
                # But must be paired with "heads = list(reversed([h for h in range(num_heads)]))" or the graph will be wrong
                data_one_task = np.flip(data_one_task, axis=1)
            except IndexError:
                print(f"No token at pos {token_idx} for prompt {prompt} of len {len(prompt)}. Setting to zeros.")
                data_one_task = np.zeros(shape=(num_heads, num_heads))

            #norm = mpl.colors.Normalize(vmin=0, vmax=np.max(data_one_task))
            im = axs[row_idx][col_idx].imshow(data_one_task, cmap=cmap, norm=norm)
            images.append(im)
            fig.colorbar(im, ax=axs[row_idx][col_idx], location='right', fraction=.1)
            axs[row_idx][col_idx].set_yticks(range(num_layers), labels=layers)
            axs[row_idx][col_idx].set_xticks(range(num_heads), labels=heads)
            axs[row_idx][col_idx].set(title=f"Tok {token_idx} '{decoded_token}'")  # formerly "token"
            axs[row_idx][col_idx].set_xlabel("Heads")
            axs[row_idx][col_idx].set_ylabel("Layers")

            # Ad-hoc label for colorbar
            axs[row_idx][col_idx].text(19.3,7,"% activity", size=10,
                           verticalalignment='center', rotation=270) 

    axs[0][0].text(3,-4,"Adj-Comparative", size=20, verticalalignment='top')
    axs[0][1].text(4,-4,"Adj-Antonym", size=20, verticalalignment='top')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.3, wspace=0, right=0.2)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Fig saved to:", save_path)
    return


def head_activity_ci_heatmap(task_info, task_pair, num_heads, num_layers, title, save_path,
                             token_positions=None):
    """
    Plot mean head activity (and optionally 95% CI width) for both tasks in task_pair.

    Args:
        task_info:       When token_positions is provided: the full tok_info dict,
                           i.e. tok_info[token_pos][task][layer][head_key] = float or dict.
                         When token_positions is None: already sliced to task level,
                           i.e. {task: {layer: {head_key: float or dict}}}.
        task_pair:       List of two task names to show as columns.
        token_positions: Optional dict mapping task name → token_pos to display,
                           e.g. {task_pair[0]: 8, task_pair[1]: 10}.
                         When None, task_info is used as-is (no token_pos slicing).
        title:           Figure suptitle.
    """
    layers = list(range(num_layers))
    heads = list(reversed(range(num_heads)))

    def _get_layer_data(task):
        if task in task_info:
            return task_info[task]
        return task_info[token_positions[task]][task]

    # Detect which optional panels are available
    sample_val = _get_layer_data(task_pair[0])[0]["head_0"]
    has_ci = isinstance(sample_val, dict)
    has_pvalue = has_ci and any(
        _get_layer_data(task)[layer][f"head_{head_idx}"].get("var_pvalue") is not None
        for task in task_pair
        for layer in range(num_layers)
        for head_idx in range(num_heads)
    )

    n_rows = (1 + int(has_ci) + int(has_pvalue))
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows), squeeze=False)

    ci_max = 0.0
    matrices = {}
    for task in task_pair:
        layer_data = _get_layer_data(task)
        mean_mat     = np.zeros((num_layers, num_heads))
        ci_width_mat = np.zeros((num_layers, num_heads))
        pvalue_mat   = np.full((num_layers, num_heads), np.nan)
        for layer in layers:
            for head_idx in range(num_heads):
                val = layer_data[layer][f"head_{head_idx}"]
                if has_ci:
                    mean_mat[layer, head_idx] = val["mean"] or 0.0
                    if val["ci_lower"] is not None and val["ci_upper"] is not None:
                        ci_width_mat[layer, head_idx] = val["ci_upper"] - val["ci_lower"]
                    if has_pvalue and val.get("var_pvalue") is not None:
                        # -log10(p): higher = more significant departure from noise
                        #pvalue_mat[layer, head_idx] = -np.log10(max(val["var_pvalue"], 1e-10))
                        pvalue_mat[layer, head_idx] = val["var_pvalue"]
                else:
                    mean_mat[layer, head_idx] = val or 0.0
                    
        # Reverse head axis to match existing heatmap convention
        matrices[task] = {
            "mean":     np.flip(mean_mat, axis=1),
            "ci_width": np.flip(ci_width_mat, axis=1),
            "pvalue":   np.flip(pvalue_mat, axis=1),
        }
        ci_max = max(ci_max, matrices[task]["ci_width"].max())

    for col_idx, task in enumerate(task_pair):
        tok_label = f" (token_pos={token_positions[task]})" if token_positions else ""
        row = 0

        im_mean = axs[row, col_idx].imshow(
            matrices[task]["mean"],
            cmap=mpl.colormaps["RdPu"],
            norm=mpl.colors.Normalize(vmin=0, vmax=1),
        )
        fig.colorbar(im_mean, ax=axs[row, col_idx], fraction=0.046, pad=0.04)
        axs[row, col_idx].set_title(f"{task}{tok_label}\nMean activity")

        if has_ci:
            row += 1
            im_ci = axs[row, col_idx].imshow(
                matrices[task]["ci_width"],
                cmap=mpl.colormaps["YlOrBr"],
                norm=mpl.colors.Normalize(vmin=0, vmax=ci_max or 1),
            )
            fig.colorbar(im_ci, ax=axs[row, col_idx], fraction=0.046, pad=0.04)
            axs[row, col_idx].set_title(f"{task}{tok_label}\n95% CI width (uncertainty)")

        if has_pvalue:
            row += 1
            pv = matrices[task]["pvalue"]
            im_pv = axs[row, col_idx].imshow(
                pv,
                cmap=mpl.colormaps["copper_r"],
                norm=mpl.colors.Normalize(vmin=0, vmax=np.nanmax(pv) or 1),
            )
            fig.colorbar(im_pv, ax=axs[row, col_idx], fraction=0.046, pad=0.04)
            axs[row, col_idx].set_title(f"{task}{tok_label}\np-value (t-test vs 0.5)")

    for ax in axs.flat:
        ax.set_yticks(range(num_layers), labels=layers)
        ax.set_xticks(range(num_heads), labels=heads)
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")

    #fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("Fig saved to:", save_path)
    return


def make_scatter(data_all_samples, save_path, title, reduction_method="pca"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Dimensionality reduction
    concat_data = np.array(data_all_samples) # shape (24, 8960)

    if reduction_method == "pca":
        pca = PCA(n_components=2)
        hs_reduced = pca.fit_transform(concat_data)
    elif reduction_method == "tsne":
        hs_reduced = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(concat_data)

    # Plot the reduced representation of each sample on one axis
    scatter = ax.scatter(hs_reduced[:, 0], hs_reduced[:, 1],
    )
    fig.suptitle(title, verticalalignment='center')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Fig saved to:", save_path)

    return


def head_k_correlation_plot(
    comparisons,
    task_pair,
    num_heads,
    num_layers,
    title,
    save_path,
    k1=1,
    k2=2,
):
    """
    Visualize how strongly each head/layer's activity profile correlates between two
    k values, for each task.

    Row 0: grouped bar chart of Spearman r per head (one bar-group per head, one bar
           per task). High r = head's activity pattern across layers is stable.
    Row 1: grouped bar chart of Spearman r per layer (one bar-group per layer).
    Row 2: per-task diff heatmaps (activity at k2 minus k1).

    Args:
        comparisons: dict mapping task_name -> output of compare_heatmaps(), i.e.
                     {task: {"spearman_per_head": [...], "spearman_per_layer": [...],
                             "diff_heatmap": np.ndarray, ...}}
        task_pair:   ordered list of task names (must be keys in comparisons)
        num_heads:   number of attention heads
        num_layers:  number of layers
        title:       figure suptitle
        save_path:   output file path
        k1, k2:      k values that were compared (used in labels only)
    """
    task_colors = ["#C72FB5", "#2F00CA", "#45BC96", "#E2CE5B"]
    n_tasks = len(task_pair)
    bar_width = 0.8 / n_tasks

    fig = plt.figure(figsize=(max(12, 5 * n_tasks), 18))
    gs = fig.add_gridspec(
        3, n_tasks,
        height_ratios=[1, 1, 1.4],
        hspace=0.5, wspace=0.45,
    )

    def _bar_chart(ax, values_by_task, x_ticks, xlabel, title_str):
        x = np.arange(len(x_ticks))
        for t_idx, task in enumerate(task_pair):
            r_vals = values_by_task[task]
            r_display = [v if not np.isnan(v) else 0.0 for v in r_vals]
            offset = (t_idx - (n_tasks - 1) / 2) * bar_width
            ax.bar(
                x + offset, r_display, width=bar_width,
                label=task, color=task_colors[t_idx % len(task_colors)],
                edgecolor="black", linewidth=0.4,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in x_ticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Spearman r  (k={k1} vs k={k2})")
        ax.set_ylim(-1.05, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_title(title_str)
        ax.legend(loc="lower right", fontsize=9)

    # Row 0: Spearman r per head
    ax_head = fig.add_subplot(gs[0, :])
    _bar_chart(
        ax_head,
        {task: comparisons[task]["spearman_per_head"] for task in task_pair},
        x_ticks=range(num_heads),
        xlabel="Head index",
        title_str=f"Per-head Spearman r  (k={k1} vs k={k2})",
    )

    # Row 1: Spearman r per layer
    ax_layer = fig.add_subplot(gs[1, :])
    _bar_chart(
        ax_layer,
        {task: comparisons[task]["spearman_per_layer"] for task in task_pair},
        x_ticks=range(num_layers),
        xlabel="Layer index",
        title_str=f"Per-layer Spearman r  (k={k1} vs k={k2})",
    )

    # Row 2: diff heatmaps per task
    layers = list(range(num_layers))
    reversed_heads = list(reversed(range(num_heads)))

    all_diffs = np.concatenate([comparisons[t]["diff_heatmap"].flatten() for t in task_pair])
    vmax = max(abs(float(all_diffs.min())), abs(float(all_diffs.max())), 1e-6)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for t_idx, task in enumerate(task_pair):
        ax = fig.add_subplot(gs[2, t_idx])
        diff_mat = np.flip(comparisons[task]["diff_heatmap"], axis=1)
        im = ax.imshow(diff_mat, cmap="RdBu_r", norm=norm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{task}\nActivity diff  (k={k2} − k={k1})")
        ax.set_yticks(range(num_layers), labels=layers)
        ax.set_xticks(range(num_heads), labels=reversed_heads)
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")

    fig.suptitle(title, y=1.01, fontsize=13)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print("Fig saved to:", save_path)
    return
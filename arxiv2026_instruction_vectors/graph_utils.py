"""
Functions for creatiing various graphs used in our analyses.
"""

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def graph_inference_scores(scores_df, instr_df, save_path, metric_type, model_names):
    # Sort models alphabetically (and by size)
    model_names = sorted(model_names)
    
    tasks = scores_df.index

    colors = [
        "#2F00CA",
        "#968CFF",  
        "#C72FB5",
        "#FDD0FF",
        "#E2CE5B",
        "#FBDE37",
        "#CE1039", 
        "#58EE60",
        "#45BC96",    
        "#68EDB3",  
        "#0055FF",
        "#9EFAFA",  
        ]
    colors = {task:color for task, color in zip(tasks, colors)}

    task_scores_row1 = {task:[] for task in tasks}
    instr_scores_row1 = {task:[] for task in tasks}
    model_names_1 = model_names[0:3]
    task_scores_row2 = {task:[] for task in tasks}
    instr_scores_row2 = {task:[] for task in tasks}
    model_names_2 = model_names[3:6]

    for task in task_scores_row1:
        for model in model_names_1:
            task_scores_row1[task].append(scores_df.loc[task, model])
            instr_scores_row1[task].append(instr_df.loc[task, model])
    for task in task_scores_row2:
        for model in model_names_2:
            task_scores_row2[task].append(scores_df.loc[task, model])
            instr_scores_row2[task].append(instr_df.loc[task, model])

    all_scores = [task_scores_row1, task_scores_row2]
    all_instr_scores = [instr_scores_row1, instr_scores_row2]
    all_models = [model_names_1, model_names_2]

    fig, axs = plt.subplots(2, 1, figsize=(4, 4), layout="constrained")

    for idx in range(len(axs)):
        ax = axs[idx]
        x = np.arange(len(all_models[idx]))
        width = 0.18  # bar width

        task_scores = all_scores[idx]
        instr_scores = all_instr_scores[idx]

        multiplier = 0
        for task, instr_score in instr_scores.items():
            offset = width * multiplier
            instr_rects = ax.bar(x + offset, instr_score, width, label=task, color="gainsboro", edgecolor="black", linewidth=0.3)
            multiplier += 1

        multiplier = 0
        for task, score in task_scores.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, score, width, label=task, color=colors[task], edgecolor="black", linewidth=0.3)
            multiplier += 1

        shortened_names = [name.replace("instruct", "it") for name in all_models[idx]]

        ax.axhline(y=50, color="gray", linestyle="dashed")
        
        ax.set_xticks(x + 0.3, shortened_names)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.yaxis.minorticks_on()

    # The axes share a legend
    handles, labels = axs[0].get_legend_handles_labels()
    print(handles)
    print(labels)
    fig.legend(handles[-4:], labels[-4:], loc='outside lower center', ncols=2)
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
            img = axs[row_idx, col_idx].imshow(values_matrices[row_idx, col_idx], cmap=cmap, norm=norm)
            images.append(img)
            fig.colorbar(img, ax=axs[row_idx, col_idx], location='right', fraction=0.046, pad=0.04, norm=norm)

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
                    #print("Avg number of changes across all samples", head_changes_all[src_token][layer])
                    #values_matrix[layer][head_idx] += 1
                    values_matrix[layer][head_idx] += head_changes_all[src_token][layer][head_idx]

        # Reverse the y-axis to make graph more readable
        values_matrix = [list(reversed(l)) for l in values_matrix]
        
        #print("MAX 1:", max(values_matrix))
        max_val = max(max(values_matrix))
        values_matrices.append(values_matrix)
        max_all_tokens.append(max_val)
        #print("MAX OF THIS:", values_matrix)

    # Make a colorbar
    cmap = mpl.colormaps['GnBu']
    flattened = np.array(values_matrices).flatten()
    max_n_changes = np.max(flattened)

    # Make it discrete normalized only if doing 1 sample (not avg)
    if max_n_changes > 1:
        bounds = np.arange(0, max_n_changes + 1)
    else:
        bounds = [0,1]
    #norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
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
                decoded_token = prompt[token_idx]
                data_one_task = np.array(both_tasks_per_token[token_idx][col_idx])
                data_one_task = data_one_task[:, :, 0]   # slice idx 0 to just get the activity % (inactivty at idx 1 follows trivially)

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
            

    #axs[0][0].text(3,-4,"Adj-Comparative", size=20, verticalalignment='top')
    #axs[0][1].text(4,-4,"Adj-Antonym", size=20, verticalalignment='top')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.3, wspace=0.4, right=0.2)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Fig saved to:", save_path)
    #plt.show()
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
    #bounds = [0,1.0]
    #norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
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
                decoded_token = prompt[token_idx]
                data_one_task = np.array(both_tasks_per_token[token_idx][col_idx])
                #norm = mpl.colors.Normalize(vmin=0, vmax=np.max(data_one_task))
                data_one_task = data_one_task[:, :, 0]   # slice idx 0 to just get the activity % (inactivty at idx 1 follows trivially)

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
    #c=color_per_task[task_name],
    #cmap=cmap_per_task[task_name],
    )
    fig.suptitle(title, verticalalignment='center')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Fig saved to:", save_path)

    return
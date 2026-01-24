import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def show_path_contributions(filtered_path_data, model, prompt, start_pos, tokenizer, save_path="path_contributions.svg"):
 
    enc = tokenizer(prompt, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    num_layers = model.model.config.num_hidden_layers

    paths_with_ranks = []

    for src_token, path, logit, rank in filtered_path_data:
        simplified_path = [(layer, token) for layer, metadata, token in path]
        paths_with_ranks.append((simplified_path, rank))

    target_idx = start_pos
    last_token_ix = len(tokens) - 1
    title = f"Attention Path Contributions from '{tokens[start_pos]}'"

    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

    n_tokens = len(tokens)
    n_layers = num_layers + 1  # Including embedding layer

    cell_width = 1.2
    cell_height = 1.0

    ranks = [rank for _, rank in paths_with_ranks]
    max_rank = max(ranks) if ranks else 1
    min_rank = min(ranks) if ranks else 0
    rank_range = max_rank - min_rank if max_rank > min_rank else 1

    for i in range(n_tokens):
        for j in range(n_layers):
            if target_idx is not None and i == target_idx:
                facecolor = '#e3f2fd' 
                edgecolor = '#1976d2'  
                linewidth = 1.0
            else:
                facecolor = '#f8f9fa'
                edgecolor = '#dee2e6'
                linewidth = 0.5

            rect = FancyBboxPatch(
                (i * cell_width, j * cell_height),
                cell_width, cell_height,
                boxstyle="round,pad=0.02",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth
            )
            ax.add_patch(rect)

    for i, token in enumerate(tokens):
        display_token = token.replace('▁', ' ').strip()
        if len(display_token) > 8:
            display_token = display_token[:8] + '...'

        fontweight = 'bold' if target_idx is not None and i == target_idx else 'normal'
        fontsize = 10 if target_idx is not None and i == target_idx else 9
        color = '#1976d2' if target_idx is not None and i == target_idx else 'black'

        ax.text(i * cell_width + cell_width/2, -0.3, display_token,
                ha='center', va='top', fontsize=fontsize, fontweight=fontweight,
                rotation=0, color=color)

    for j in range(n_layers):
        if j == 0:
            layer_label = "Emb"
        else:
            layer_label = f"L{j-1}"
        ax.text(-0.3, j * cell_height + cell_height/2, layer_label,
                ha='right', va='center', fontsize=9)

    for path, rank in paths_with_ranks:
        points = []
        for layer_idx, token_idx in path:
            layer_plot = layer_idx + 1 if layer_idx >= 0 else 0
            x = token_idx * cell_width + cell_width/2
            y = layer_plot * cell_height + cell_height/2
            points.append((x, y))

        normalized_rank = (rank - min_rank) / rank_range if rank_range > 0 else 0.5

        alpha = 0.8 - 0.6 * normalized_rank  
        linewidth = 2.5 - 1.5 * normalized_rank  

        color = plt.cm.Blues(0.9 - 0.5 * normalized_rank) 
        zorder = int(1000 * (1 - normalized_rank)) 

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-',
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                connectionstyle="arc3,rad=0.1",
                zorder=zorder
            )
            ax.add_patch(arrow)

    if target_idx is not None:
        source_marker = plt.Circle(
            (target_idx * cell_width + cell_width/2, cell_height/2),
            0.15, color='#1976d2', alpha=1.0, zorder=1000
        )
        ax.add_patch(source_marker)

    final_rect = FancyBboxPatch(
        (last_token_ix * cell_width + 0.1, num_layers * cell_height + 0.1),
        cell_width - 0.2, cell_height - 0.2,
        boxstyle="round,pad=0.02",
        facecolor='#4caf50',
        edgecolor='#2e7d32',
        linewidth=2,
        alpha=0.7,
        zorder=1000
    )
    ax.add_patch(final_rect)

    legend_elements = [
        patches.Patch(color=plt.cm.Blues(0.9), alpha=0.8, label='High contribution (low rank)'),
        patches.Patch(color=plt.cm.Blues(0.65), alpha=0.5, label='Medium contribution'),
        patches.Patch(color=plt.cm.Blues(0.4), alpha=0.2, label='Low contribution (high rank)'),
    ]
    if target_idx is not None:
        legend_elements.append(patches.Circle((0, 0), 0.1, color='#1976d2', label='Source token'))
    legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor='#4caf50', alpha=0.7, label='Target token'))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.set_xlim(-0.5, n_tokens * cell_width + 0.5)
    ax.set_ylim(-0.5, n_layers * cell_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    unique_paths = len(set(tuple(path) for path, _ in paths_with_ranks))
    plt.title(f"{title}\n{len(paths_with_ranks)} paths, {unique_paths} unique",
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    plt.savefig(save_path.replace('.svg', '.png'), format='png', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Visualization saved as {save_path} and {save_path.replace('.svg', '.png')}")

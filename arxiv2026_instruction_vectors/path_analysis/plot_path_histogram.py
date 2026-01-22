import numpy as np
import matplotlib.pyplot as plt


def plot_path_histogram(answer_str, path_data, hs_save_path):
    if not path_data:
        print("Warning: No path data to plot")
        return

    print(f"\nPlotting histograms for {len(path_data)} paths...")

    path_logits = [logit for _, _, logit, _ in path_data]
    path_ranks = [rank for _, _, _, rank in path_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(path_logits, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Logit Contribution')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Path Logit Contributions to "{answer_str}"')
    ax1.grid(True, alpha=0.3)

    ax2.hist(path_ranks, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of Answer Token Ranks Across Paths')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(hs_save_path, dpi=150, bbox_inches='tight')
    print(f"Histograms saved to: ", hs_save_path)

    print(f"\nPathwise Logit statistics:")
    print(f"  Mean: {np.mean(path_logits):.4f}")
    print(f"  Median: {np.median(path_logits):.4f}")
    print(f"  Std: {np.std(path_logits):.4f}")
    print(f"  Min: {np.min(path_logits):.4f}, Max: {np.max(path_logits):.4f}")

    print(f"\nPathwise Rank statistics:")
    print(f"  Mean: {np.mean(path_ranks):.2f}")
    print(f"  Median: {np.median(path_ranks):.2f}")
    print(f"  Std: {np.std(path_ranks):.2f}")
    print(f"  Min: {np.min(path_ranks)}, Max: {np.max(path_ranks)}")

    top1 = sum(1 for r in path_ranks if r == 1)
    top10 = sum(1 for r in path_ranks if r <= 10)
    top100 = sum(1 for r in path_ranks if r <= 100)
    top1000 = sum(1 for r in path_ranks if r <= 1000)

    print(f"\nRank distribution:")
    print(f"  Rank 1: {top1} paths ({100*top1/len(path_ranks):.1f}%)")
    print(f"  Top 10: {top10} paths ({100*top10/len(path_ranks):.1f}%)")
    print(f"  Top 100: {top100} paths ({100*top100/len(path_ranks):.1f}%)")
    print(f"  Top 1000: {top1000} paths ({100*top1000/len(path_ranks):.1f}%)")
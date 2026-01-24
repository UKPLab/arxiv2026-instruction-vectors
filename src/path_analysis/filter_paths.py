def filter_paths_by_threshold(path_data, threshold_rank=None, threshold_logit=None):
    if threshold_rank is not None:
        filtered = list(filter(lambda x: x[3] <= threshold_rank, path_data))
        print(f"\nFiltering by rank <= {threshold_rank}: {len(filtered)}/{len(path_data)} paths kept ({100*len(filtered)/len(path_data):.1f}%)")
        return filtered
    elif threshold_logit is not None:
        filtered = list(filter(lambda x: x[2] >= threshold_logit, path_data))
        print(f"\nFiltering by logit >= {threshold_logit}: {len(filtered)}/{len(path_data)} paths kept ({100*len(filtered)/len(path_data):.1f}%)")
        return filtered
    else:
        return path_data

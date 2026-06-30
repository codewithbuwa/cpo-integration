# Cluster assignment for CPO.
#
# Pluggable layer: produces a `prompt_id -> cluster_id` mapping that the
# dataset loaders in train/data.py can call when constructing Example
# instances. Once the cluster source is decided (sidecar JSON, dataset
# field, online embedding-based, etc.), only this file needs to grow.

import json
from typing import Dict, Optional

from .utils import rank0_print


def load_cluster_map(
    path: Optional[str], num_clusters: int, num_clusters_check: bool = True
) -> Dict[str, int]:
    """Load a {prompt_id: cluster_id} mapping from a sidecar JSON file.

    Args:
        path: path to a JSON file mapping prompt_id (hex string) -> int in [0, K).
              If None, returns an empty dict (caller falls back to cluster 0).
        num_clusters: K, used to validate values in the file.
        num_clusters_check: if True, assert all values are in [0, K).

    Returns:
        Dict[str, int]. Missing prompt_ids should be defaulted to 0 by the caller.
    """
    if path is None:
        rank0_print("No cluster_map_path set; all examples default to cluster 0 "
                    "(equivalent to KTO with global z_0).")
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    out = {str(k): int(v) for k, v in raw.items()}

    if num_clusters_check:
        bad = [v for v in out.values() if not (0 <= v < num_clusters)]
        if bad:
            raise ValueError(
                f"cluster_map at {path} has {len(bad)} values outside [0, {num_clusters}); "
                f"first few: {bad[:5]}"
            )

    rank0_print(f"Loaded cluster_map ({len(out)} prompts, K={num_clusters}) from {path}")
    return out


def assign_cluster(prompt_id: str, cluster_map: Dict[str, int], default: int = 0) -> int:
    """Look up cluster_id for a prompt; default to 0 if unmapped."""
    return cluster_map.get(str(prompt_id), default)

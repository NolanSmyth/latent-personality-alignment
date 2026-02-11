"""Centralized HuggingFace cache path resolution."""

import os
import glob


def get_cache_dir() -> str:
    """
    Get the HuggingFace cache directory.

    Priority:
    1. HF_HOME environment variable
    2. HF_HUB_CACHE environment variable
    3. TRANSFORMERS_CACHE environment variable
    4. Default: /scratch/$USER/.cache/huggingface (cluster)
    5. Fallback: ~/.cache/huggingface (local)
    """
    for env_var in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        if os.environ.get(env_var):
            return os.environ[env_var]

    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    scratch_cache = f"/scratch/{user}/.cache/huggingface"
    if os.path.isdir("/scratch"):
        return scratch_cache

    return os.path.expanduser("~/.cache/huggingface")


def get_model_path(model_name: str) -> str:
    """
    Resolve a HuggingFace model name to a local path.

    Checks (in order):
    1. Direct path (already a local directory)
    2. HF cache snapshot (hub/models--org--name/snapshots/*)

    Returns the resolved path (usable directly in from_pretrained).
    Raises FileNotFoundError with diagnostic info if not found.
    """
    # Already a local path
    if os.path.isdir(model_name):
        return model_name

    # Standard HF cache layout: hub/models--{org}--{name}/snapshots/{hash}
    cache_dir = get_cache_dir()
    model_dir_name = model_name.replace("/", "--")
    snapshot_pattern = f"{cache_dir}/hub/models--{model_dir_name}/snapshots/*"
    snapshots = sorted(glob.glob(snapshot_pattern))

    if snapshots:
        return snapshots[-1]

    raise FileNotFoundError(
        f"Model '{model_name}' not found in local cache.\n"
        f"  Searched: {snapshot_pattern}\n"
        f"  Cache dir: {cache_dir}\n"
        f"  Set HF_HOME or download the model first."
    )

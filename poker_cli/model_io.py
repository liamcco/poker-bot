from __future__ import annotations

import glob
import os
import re
from typing import Optional, Tuple

import torch

from poker_ml import PolicyNet


def load_or_init_model(checkpoint_path: str, hidden: int, device: torch.device) -> Tuple[PolicyNet, str]:
    """
    Load a checkpoint if available; otherwise return a random initialized model.

    The fallback behavior keeps the UI usable even before any training run exists.
    """
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            model_hidden = int(ckpt_args.get("hidden", hidden))
            model = PolicyNet(hidden=model_hidden).to(device)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.eval()
            return model, f"Loaded model checkpoint: {checkpoint_path}"
        except Exception as e:
            model = PolicyNet(hidden=hidden).to(device)
            model.eval()
            return (
                model,
                "Found checkpoint but could not load it cleanly "
                f"({e.__class__.__name__}: {e}). Using random untrained model.",
            )

    model = PolicyNet(hidden=hidden).to(device)
    model.eval()
    return model, f"No checkpoint found at {checkpoint_path}; using random untrained model."


def _checkpoint_episode(path: str) -> int:
    """Extract episode number from checkpoint filename for sorting."""
    m = re.search(r"checkpoint_ep(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_model_path(model_path: Optional[str], out_dir: str) -> Optional[str]:
    """
    Resolve which model file to load.

    Priority:
    1. Explicit --model-path
    2. Latest checkpoint_ep*.pt in out_dir
    3. final_model.pt in out_dir
    """
    if model_path:
        return model_path

    pattern = os.path.join(out_dir, "checkpoint_ep*.pt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        checkpoints.sort(key=_checkpoint_episode)
        return checkpoints[-1]

    final_path = os.path.join(out_dir, "final_model.pt")
    if os.path.exists(final_path):
        return final_path
    return None

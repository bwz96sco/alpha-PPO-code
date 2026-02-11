from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alphasched.search.policy import Policy


@dataclass(slots=True)
class SB3MaskablePolicy(Policy):
    """SB3 `MaskablePPO` wrapper exposing action probabilities for search."""

    model_path: Path
    device: str = "cpu"
    name: str = "sb3_maskable_ppo"
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # lazy import: keeps core tests usable without SB3 installed
        from sb3_contrib import MaskablePPO  # type: ignore

        self._model = MaskablePPO.load(str(self.model_path), device=self.device)

    def action_probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        import torch as th

        model = self._model
        obs_t = th.as_tensor(obs[None], device=model.device).float()
        mask_t = th.as_tensor(action_mask[None], device=model.device).bool()

        with th.no_grad():
            dist = model.policy.get_distribution(obs_t, action_masks=mask_t)
            torch_dist = getattr(dist, "distribution", dist)
            probs = getattr(torch_dist, "probs", None)
            if probs is None:
                logits = torch_dist.logits  # type: ignore[attr-defined]
                probs = th.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy().squeeze(0)

from typing import Optional

import torch

from ._ops import ops


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache,
                         is_neox)

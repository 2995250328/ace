# Copyright © Niantic, Inc. 2022.

import logging
from contextlib import contextmanager

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

_logger = logging.getLogger(__name__)


@contextmanager
def _progress(desc, total):
    if tqdm is None:
        yield None
        return

    with tqdm(total=total, desc=desc, unit="step", leave=False) as bar:
        yield bar


def get_pixel_grid(subsampling_factor, *, show_progress=False, desc="Building pixel grid"):
    """Generate target pixel positions according to a subsampling factor."""

    if show_progress:
        _logger.info(f"{desc}...")

    def _build_grid(progress):
        pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
        if progress is not None:
            progress.update(1)

        yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij')
        if progress is not None:
            progress.update(1)

        grid = subsampling_factor * (torch.stack([xx, yy]) + 0.5)
        if progress is not None:
            progress.update(1)

        return grid

    if show_progress:
        with _progress(desc, 3) as progress:
            grid = _build_grid(progress)
        _logger.info(f"{desc} complete.")
        return grid

    return _build_grid(None)


def to_homogeneous(input_tensor, dim=1):
    """Converts tensor to homogeneous coordinates by adding ones to the specified dimension."""

    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output

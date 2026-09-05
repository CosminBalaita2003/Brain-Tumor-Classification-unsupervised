# feature_extraction_hog.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog


PathLike = Union[str, Path]


def _flatten_paths_deep(obj) -> List[str]:

    out: List[str] = []

    def rec(x):
        if x is None:
            return
        if isinstance(x, (str, Path)):
            out.append(str(x))
            return
        if isinstance(x, (list, tuple, np.ndarray)):
            for y in x:
                rec(y)
            return
        out.append(str(x))

    rec(obj)
    return out


def _read_grayscale(path: str) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3:

        img = np.asarray(img[..., :3], dtype=np.float64)
        if img.max() > 1.5:
            img /= 255.0
        img = (
            0.2125 * img[..., 0]
            + 0.7154 * img[..., 1]
            + 0.0721 * img[..., 2]
        )
    else:
        img = img.astype(np.float64)
        if img.max() > 1.5:
            img = img / 255.0
    img = np.asarray(img, dtype=np.float64)
    if not np.all(np.isfinite(img)):
        raise ValueError(f"Image contains non-finite values: {path}")
    return np.clip(img, 0.0, 1.0)


def extract_features(
    image_paths,
    image_size: Tuple[int, int] = (128, 128),
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    block_norm: str = "L2-Hys",
    skip_errors: bool = True,
) -> np.ndarray:

    paths = _flatten_paths_deep(image_paths)

    feats = []
    kept_paths = 0
    for p in paths:
        try:
            img = _read_grayscale(p)
            if image_size is not None:
                img = resize(
                    img, image_size, anti_aliasing=True, preserve_range=True
                ).astype(np.float64)
                if img.max() > 1.5:
                    img = img / 255.0

            f = hog(
                img,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm=block_norm,
                feature_vector=True,
            )
            feats.append(f.astype(np.float32))
            kept_paths += 1
        except Exception as e:
            if not skip_errors:
                raise
            continue

    if len(feats) == 0:
        raise RuntimeError(
            "HOG extract_features: no features extracted. "
            "Check paths / image formats. (All files may have failed to load.)"
        )

    X = np.vstack(feats).astype(np.float32)
    return X

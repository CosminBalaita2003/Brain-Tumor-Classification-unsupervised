# feature_extraction_handcrafted.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


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
        img = rgb2gray(img)  # float [0,1]
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
    img = np.asarray(img, dtype=np.float32)
    return img


def _safe_skew(x: np.ndarray) -> float:
    # skew = E[(x-mu)^3] / sigma^3
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return 0.0
    return float(np.mean((x - mu) ** 3) / (sigma ** 3 + 1e-12))


def _safe_kurtosis(x: np.ndarray) -> float:
    # kurtosis (not excess) = E[(x-mu)^4] / sigma^4
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return 0.0
    return float(np.mean((x - mu) ** 4) / (sigma ** 4 + 1e-12))


def extract_features(
    image_paths,
    image_size: Tuple[int, int] = (128, 128),
    hist_bins: int = 32,
    lbp_P: int = 8,
    lbp_R: int = 1,
    glcm_distances=(1, 2),
    glcm_angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    skip_errors: bool = True,
) -> np.ndarray:

    paths = _flatten_paths_deep(image_paths)

    all_feats = []
    for p in paths:
        try:
            img = _read_grayscale(p)

            if image_size is not None:
                img = resize(img, image_size, anti_aliasing=True, preserve_range=True).astype(np.float32)
                if img.max() > 1.5:
                    img = img / 255.0

            # Flatten pixels in [0,1]
            pix = img.flatten().astype(np.float32)

            feats = []

            # 1) basic stats
            feats.append(float(np.mean(pix)))
            feats.append(float(np.std(pix)))
            feats.append(_safe_skew(pix))
            feats.append(_safe_kurtosis(pix))

            # 2) histogram (normalized)
            hist, _ = np.histogram(pix, bins=hist_bins, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-12)
            feats.extend(hist.tolist())

            # 3) LBP histogram (uniform) — use uint8 to avoid float warnings
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            lbp = local_binary_pattern(img_u8, P=lbp_P, R=lbp_R, method="uniform")

            lbp_bins = lbp_P + 2
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=lbp_bins, range=(0, lbp_bins))
            feats.extend(lbp_hist.tolist())

            # 4) GLCM props
            # GLCM expects integer levels; quantize to 8-bit
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            glcm = graycomatrix(
                img_u8,
                distances=list(glcm_distances),
                angles=list(glcm_angles),
                levels=256,
                symmetric=True,
                normed=True,
            )

            props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
            for pr in props:
                v = graycoprops(glcm, pr)  # shape (len(dist), len(angle))
                feats.append(float(np.mean(v)))

            all_feats.append(np.asarray(feats, dtype=np.float32))

        except Exception:
            if not skip_errors:
                raise
            continue

    if len(all_feats) == 0:
        raise RuntimeError(
            "Handcrafted extract_features: no features extracted. "
            "Check paths / image formats. (All files may have failed to load.)"
        )

    X = np.vstack(all_feats).astype(np.float32)
    return X

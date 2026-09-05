"""Shared loading, evaluation, and plotting helpers for clustering experiments."""

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)

from feature_extraction_handcrafted import extract_features as extract_handcrafted
from feature_extraction_hog import extract_features as extract_hog


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_metadata(splits_dir: Path):
    metadata = json.loads((splits_dir / "labels.json").read_text(encoding="utf-8"))
    return metadata["classes"], metadata["class_to_label"]


def load_split(splits_dir: Path, split: str, classes, class_to_label):
    paths, labels = [], []
    for class_name in classes:
        images = sorted(
            p for p in (splits_dir / split / class_name).glob("*")
            if p.suffix.lower() in EXTENSIONS
        )
        paths.extend(str(path) for path in images)
        labels.extend([class_to_label[class_name]] * len(images))
    return paths, np.asarray(labels, dtype=int)


def extract(feature_name: str, paths):
    extractors = {"hog": extract_hog, "handcrafted": extract_handcrafted}
    if feature_name not in extractors:
        raise ValueError("feature must be 'hog' or 'handcrafted'")
    features = np.asarray(extractors[feature_name](paths), dtype=np.float64)
    if features.shape[0] != len(paths):
        raise ValueError(
            "Feature extraction skipped one or more images; labels would no "
            "longer align with samples. Re-run with valid image files."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Extracted features contain NaN or infinite values.")
    return features


def safe_pca_transform(pca, features):
    """Apply a fitted PCA without NumPy/BLAS matrix multiplication.

    Some macOS Python 3.9 environments used by this repository produce
    spurious overflow values for otherwise finite arrays when ``@`` is used.
    ``einsum(..., optimize=False)`` computes the identical projection using
    direct contractions and avoids that broken backend.
    """
    values = np.asarray(features, dtype=np.float64)
    centred = values - np.asarray(pca.mean_, dtype=np.float64)
    components = np.asarray(pca.components_, dtype=np.float64)
    transformed = np.einsum("ij,kj->ik", centred, components, optimize=False)
    if not np.all(np.isfinite(transformed)):
        raise FloatingPointError("PCA projection produced non-finite values.")
    return transformed


def optimal_mapping(y_true, clusters):
    true_values = np.unique(y_true)
    cluster_values = np.unique(clusters)
    counts = np.zeros((len(cluster_values), len(true_values)), dtype=int)
    for row, cluster in enumerate(cluster_values):
        for column, label in enumerate(true_values):
            counts[row, column] = np.sum((clusters == cluster) & (y_true == label))
    rows, columns = linear_sum_assignment(-counts)
    mapping = {int(cluster_values[r]): int(true_values[c]) for r, c in zip(rows, columns)}
    majority = int(np.bincount(y_true).argmax())
    return mapping, majority


def evaluate(y_true, clusters, embedding=None):
    mapping, majority = optimal_mapping(y_true, clusters)
    mapped = np.asarray([mapping.get(int(c), majority) for c in clusters], dtype=int)
    non_noise = clusters != -1
    silhouette = None
    if embedding is not None and np.unique(clusters[non_noise]).size > 1:
        silhouette = float(silhouette_score(embedding[non_noise], clusters[non_noise]))
    return {
        "ARI": float(adjusted_rand_score(y_true, clusters)),
        "NMI": float(normalized_mutual_info_score(y_true, clusters)),
        "mapped_accuracy": float(np.mean(mapped == y_true)),
        "silhouette": silhouette,
        "mapping": mapping,
    }, mapped


def save_confusion(out_path: Path, y_true, mapped, classes, title: str):
    matrix = confusion_matrix(y_true, mapped, labels=range(len(classes)))
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)))
    axis.set_xticklabels(classes, rotation=20, ha="right")
    axis.set_yticklabels(classes)
    axis.set_xlabel("Mapped cluster")
    axis.set_ylabel("True class")
    axis.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_representatives(out_dir: Path, paths, embedding, clusters, top_k=8):
    root = out_dir / "representatives"
    root.mkdir(parents=True, exist_ok=True)
    for cluster in np.unique(clusters):
        indices = np.where(clusters == cluster)[0]
        if not len(indices):
            continue
        centre = embedding[indices].mean(axis=0)
        order = indices[np.argsort(np.linalg.norm(embedding[indices] - centre, axis=1))]
        cluster_dir = root / ("noise" if cluster == -1 else f"cluster_{cluster}")
        cluster_dir.mkdir(exist_ok=True)
        for rank, index in enumerate(order[:top_k], start=1):
            source = Path(paths[int(index)])
            shutil.copy2(source, cluster_dir / f"{rank:02d}_{source.name}")

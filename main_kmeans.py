"""Binary KMeans clustering with validation-only hyperparameter selection."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clustering_utils import (
    evaluate, extract, load_metadata, load_split, save_confusion,
    save_representatives, safe_pca_transform,
)


PCA_GRID = [10, 20, 30, 50, 80, 100]
INIT_GRID = ["k-means++", "random"]
N_INIT_GRID = [10, 20, 40]
SEED = 42


def main(feature: str):
    splits = Path("splits_brain_tumor")
    classes, mapping = load_metadata(splits)
    data = {name: load_split(splits, name, classes, mapping) for name in ("train", "val", "test")}
    features = {name: extract(feature, paths) for name, (paths, _) in data.items()}
    out_dir = Path("outputs") / "kmeans" / feature
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler().fit(features["train"])
    scaled = {name: scaler.transform(values) for name, values in features.items()}
    maximum = min(scaled["train"].shape) - 1
    dimensions = [d for d in PCA_GRID if d <= maximum] or [maximum]
    results, candidates = [], []

    for dimension in dimensions:
        # ARPACK avoids NumPy's randomized matrix-multiplication path, which
        # emits overflow warnings with the macOS/Python 3.9 environment used
        # by this project.
        pca = PCA(
            n_components=dimension, svd_solver="arpack", random_state=SEED
        ).fit(scaled["train"])
        embedded = {
            name: safe_pca_transform(pca, values)
            for name, values in scaled.items()
        }
        for init in INIT_GRID:
            for n_init in N_INIT_GRID:
                model = KMeans(
                    n_clusters=2, init=init, n_init=n_init,
                    max_iter=500, random_state=SEED,
                ).fit(embedded["train"])
                validation_clusters = model.predict(embedded["val"])
                metrics, _ = evaluate(data["val"][1], validation_clusters, embedded["val"])
                row = {
                    "params": {"pca_dim": dimension, "init": init, "n_init": n_init},
                    "validation": metrics,
                }
                results.append(row)
                candidates.append((metrics["NMI"], row, model, pca, embedded))

    _, best, model, pca, embedded = max(candidates, key=lambda item: item[0])
    validation_clusters = model.predict(embedded["val"])
    test_clusters = model.predict(embedded["test"])
    validation_metrics, validation_mapped = evaluate(data["val"][1], validation_clusters, embedded["val"])
    test_metrics, test_mapped = evaluate(data["test"][1], test_clusters, embedded["test"])
    save_confusion(out_dir / "confusion_matrix_val.png", data["val"][1], validation_mapped, classes, "KMeans validation")
    save_confusion(out_dir / "confusion_matrix_test.png", data["test"][1], test_mapped, classes, "KMeans test")
    save_representatives(out_dir, data["test"][0], embedded["test"], test_clusters)

    fig, axis = plt.subplots()
    for init in INIT_GRID:
        xs, ys = [], []
        for dimension in dimensions:
            subset = [r for r in results if r["params"]["pca_dim"] == dimension and r["params"]["init"] == init]
            xs.append(dimension)
            ys.append(max(r["validation"]["NMI"] for r in subset))
        axis.plot(xs, ys, marker="o", label=init)
    axis.set(xlabel="PCA components", ylabel="Best validation NMI", title="KMeans grid search")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "validation_nmi_vs_pca.png", dpi=160)
    plt.close(fig)

    output = {
        "model": "KMeans", "feature": feature, "classes": classes,
        "selection_metric": "validation NMI", "best_params": best["params"],
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "grid_results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("best_params", "validation_metrics", "test_metrics")}, indent=2))


if __name__ == "__main__":
    for feature_name in ("hog", "handcrafted"):
        print(f"\n{'=' * 72}\nRunning KMeans with {feature_name} features\n{'=' * 72}")
        main(feature_name)

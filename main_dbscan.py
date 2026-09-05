
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clustering_utils import (
    evaluate, extract, load_metadata, load_split, save_confusion,
    save_representatives, safe_pca_transform,
)


PCA_GRID = [10, 20, 30, 50, 80, 100]
METRIC_GRID = ["euclidean", "cosine"]
EPS_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
MIN_SAMPLES_GRID = [3, 5, 10, 15, 20]
SEED = 42


def cluster_count(labels):
    return len(set(labels) - {-1})


def selection_score(metrics, labels):
    noise_fraction = float(np.mean(labels == -1))
    return metrics["NMI"] - 0.05 * abs(cluster_count(labels) - 2) - 0.05 * noise_fraction


def main(feature: str):
    splits = Path("splits_brain_tumor")
    classes, mapping = load_metadata(splits)
    data = {name: load_split(splits, name, classes, mapping) for name in ("train", "val", "test")}
    features = {name: extract(feature, paths) for name, (paths, _) in data.items()}
    out_dir = Path("outputs") / "dbscan" / feature
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler().fit(features["train"])
    scaled = {name: scaler.transform(values) for name, values in features.items()}
    maximum = min(scaled["train"].shape) - 1
    dimensions = [d for d in PCA_GRID if d <= maximum] or [maximum]
    results, candidates = [], []

    for dimension in dimensions:
        pca = PCA(
            n_components=dimension, svd_solver="arpack", random_state=SEED
        ).fit(scaled["train"])
        embedded = {
            name: safe_pca_transform(pca, values)
            for name, values in scaled.items()
        }
        for metric in METRIC_GRID:
            for eps in EPS_GRID:
                for min_samples in MIN_SAMPLES_GRID:
                    params = {"pca_dim": dimension, "metric": metric, "eps": eps, "min_samples": min_samples}
                    labels = DBSCAN(
                        eps=eps, min_samples=min_samples, metric=metric, n_jobs=1
                    ).fit_predict(embedded["val"])
                    metrics, _ = evaluate(data["val"][1], labels, embedded["val"])
                    score = selection_score(metrics, labels)
                    row = {
                        "params": params, "validation": metrics,
                        "clusters": cluster_count(labels), "noise_points": int(np.sum(labels == -1)),
                        "selection_score": score,
                    }
                    results.append(row)
                    candidates.append((score, row, pca, embedded))

    _, best, pca, embedded = max(candidates, key=lambda item: item[0])
    params = {key: value for key, value in best["params"].items() if key != "pca_dim"}
    validation_clusters = DBSCAN(**params, n_jobs=1).fit_predict(embedded["val"])
    test_clusters = DBSCAN(**params, n_jobs=1).fit_predict(embedded["test"])
    validation_metrics, validation_mapped = evaluate(data["val"][1], validation_clusters, embedded["val"])
    test_metrics, test_mapped = evaluate(data["test"][1], test_clusters, embedded["test"])
    save_confusion(out_dir / "confusion_matrix_val.png", data["val"][1], validation_mapped, classes, "DBSCAN validation")
    save_confusion(out_dir / "confusion_matrix_test.png", data["test"][1], test_mapped, classes, "DBSCAN test")
    save_representatives(out_dir, data["test"][0], embedded["test"], test_clusters)

    fig, axis = plt.subplots()
    for metric in METRIC_GRID:
        xs, ys = [], []
        for eps in EPS_GRID:
            subset = [r for r in results if r["params"]["metric"] == metric and r["params"]["eps"] == eps]
            xs.append(eps)
            ys.append(max(r["validation"]["NMI"] for r in subset))
        axis.plot(xs, ys, marker="o", label=metric)
    axis.set(xlabel="eps", ylabel="Best validation NMI", title="DBSCAN grid search")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "validation_nmi_vs_eps.png", dpi=160)
    plt.close(fig)

    output = {
        "model": "DBSCAN", "feature": feature, "classes": classes,
        "selection_metric": "validation NMI with cluster/noise penalty",
        "best_params": best["params"], "validation_metrics": validation_metrics,
        "test_metrics": test_metrics, "validation_clusters": cluster_count(validation_clusters),
        "test_clusters": cluster_count(test_clusters), "test_noise_points": int(np.sum(test_clusters == -1)),
        "grid_results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("best_params", "validation_metrics", "test_metrics")}, indent=2))


if __name__ == "__main__":
    for feature_name in ("hog", "handcrafted"):
        print(f"\n{'=' * 72}\nRunning DBSCAN with {feature_name} features\n{'=' * 72}")
        main(feature_name)

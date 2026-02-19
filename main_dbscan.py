# main_dbscan.py
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from feature_extraction_hog import extract_features as extract_hog
from feature_extraction_handcrafted import extract_features as extract_hand



SPLITS_DIR = "splits_brain_tumor"
MODEL_NAME = "dbscan"
FEATURE_NAME = "hog"  # "hog" or "handcrafted"
RANDOM_STATE = 42

# PCA grid (auto-clipped to <= min(n_samples, n_features))
GRID_PCA = [10, 20, 30, 50, 80, 100, 150, 200]

# DBSCAN grid
GRID_METRIC = ["euclidean", "cosine"]
GRID_EPS = [0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
GRID_MIN_SAMPLES = [3, 5, 10, 15, 20]

# Selection
TARGET_CLUSTERS = 4
MIN_CLUSTERS_REQUIRED = 3

# If True, select using a "fair" score that prefers ~4 clusters and low noise.
USE_FAIR_SCORE = True
LAMBDA_CLUSTERS = 0.02   # penalty per cluster away from 4
LAMBDA_NOISE = 0.0005    # penalty per noise point

TOPK_REP = 12  # representative images per class



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_labels_json(splits_dir: Path):
    with open(splits_dir / "labels.json", "r", encoding="utf-8") as f:
        j = json.load(f)
    classes = j["classes"]
    class_to_label = {k: int(v) for k, v in j["class_to_label"].items()}
    return classes, class_to_label


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def load_split_folder(splits_dir: Path, split: str, classes, class_to_label):
    paths, y = [], []
    for cls in classes:
        cls_dir = splits_dir / split / cls
        if not cls_dir.exists():
            continue
        imgs = sorted(list_images(cls_dir))
        for p in imgs:
            paths.append(str(p))
            y.append(class_to_label[cls])
    return paths, np.asarray(y, dtype=int)


def pick_extractor(feature_name: str):
    if feature_name == "hog":
        return extract_hog
    if feature_name == "handcrafted":
        return extract_hand
    raise ValueError("FEATURE_NAME must be 'hog' or 'handcrafted'")



def hungarian_map_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if np.any(y_pred == -1):
        y_pred = y_pred.copy()
        y_pred[y_pred == -1] = y_pred.max() + 1

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    C = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    pred_to_row = {c: i for i, c in enumerate(pred_labels)}
    true_to_col = {c: j for j, c in enumerate(true_labels)}

    for t, p in zip(y_true, y_pred):
        C[pred_to_row[p], true_to_col[t]] += 1

    row_ind, col_ind = linear_sum_assignment(-C)

    mapping = {}
    correct = 0
    for r, c in zip(row_ind, col_ind):
        pred_cluster = int(pred_labels[r])
        true_label = int(true_labels[c])
        mapping[pred_cluster] = true_label
        correct += C[r, c]

    # fill missing clusters
    for c in pred_labels:
        c = int(c)
        if c not in mapping:
            mapping[c] = -1

    return float(correct / len(y_true)), mapping


def clustering_metrics(y_true, y_pred):
    hacc, mapping = hungarian_map_accuracy(y_true, y_pred)
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "HungAcc": float(hacc),
        "mapping": mapping,
    }


def count_clusters_and_noise(labels):
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len([u for u in uniq.tolist() if u != -1]))
    return n_clusters, n_noise


def selection_score(nmi: float, n_clusters: int, n_noise: int) -> float:

    if not USE_FAIR_SCORE:
        return float(nmi)
    return float(nmi) - LAMBDA_CLUSTERS * abs(int(n_clusters) - TARGET_CLUSTERS) - LAMBDA_NOISE * int(n_noise)



def plot_best_metric_vs_pca(out_dir: Path, all_val_results, metric_name="NMI"):
    pca_dims = sorted({int(r["params"]["pca_dim"]) for r in all_val_results})
    xs, ys = [], []

    for p in pca_dims:
        subset = [r for r in all_val_results if int(r["params"]["pca_dim"]) == p]
        if not subset:
            continue

        # prefer configs that satisfy min cluster constraint; fallback otherwise
        ok = [r for r in subset if int(r["val_clusters_found"]) >= MIN_CLUSTERS_REQUIRED]
        use = ok if ok else subset

        best = max(use, key=lambda r: float(r["val"][metric_name]))
        xs.append(p)
        ys.append(float(best["val"][metric_name]))

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("PCA dim")
    plt.ylabel(f"Best VAL {metric_name}")
    plt.title(f"DBSCAN: Best VAL {metric_name} vs PCA dim")
    plt.tight_layout()
    plt.savefig(out_dir / f"val_best_{metric_name.lower()}_vs_pca_dim.png", dpi=150)
    plt.close()


def plot_best_fairscore_vs_pca(out_dir: Path, all_val_results):
    if not USE_FAIR_SCORE:
        return

    pca_dims = sorted({int(r["params"]["pca_dim"]) for r in all_val_results})
    xs, ys = [], []
    for p in pca_dims:
        subset = [r for r in all_val_results if int(r["params"]["pca_dim"]) == p]
        if not subset:
            continue
        ok = [r for r in subset if int(r["val_clusters_found"]) >= MIN_CLUSTERS_REQUIRED]
        use = ok if ok else subset
        best = max(use, key=lambda r: float(r["val_selection_score"]))
        xs.append(p)
        ys.append(float(best["val_selection_score"]))

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("PCA dim")
    plt.ylabel("Best VAL selection score")
    plt.title("DBSCAN: Best VAL selection score vs PCA dim")
    plt.tight_layout()
    plt.savefig(out_dir / "val_best_selection_score_vs_pca_dim.png", dpi=150)
    plt.close()


def save_representatives_dbscan(
    out_dir: Path,
    img_paths,
    X_embed,          # embedding (PCA space)
    y_true,
    y_pred,           # DBSCAN labels (-1 = noise)
    n_classes: int,
    topk: int = 12,
):

    reps_root = out_dir / "representatives"
    ensure_dir(reps_root)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    X = np.asarray(X_embed, dtype=np.float32)

    used_paths = set()

    # precompute centroids for each non-noise cluster
    centroids = {}
    for k in np.unique(y_pred):
        k = int(k)
        if k == -1:
            continue
        idx_k = np.where(y_pred == k)[0]
        if len(idx_k) == 0:
            continue
        centroids[k] = X[idx_k].mean(axis=0)

    for t in range(n_classes):
        class_dir = reps_root / f"class_{t}"
        ensure_dir(class_dir)

        idx_t = np.where(y_true == t)[0]
        if len(idx_t) == 0:
            continue

        # consider only non-noise points
        idx_t_non_noise = idx_t[y_pred[idx_t] != -1]
        best_k = None
        best_cnt = 0

        if len(idx_t_non_noise) > 0:
            clusters, counts = np.unique(y_pred[idx_t_non_noise], return_counts=True)
            # clusters here are non-noise already
            best_k = int(clusters[np.argmax(counts)])
            best_cnt = int(np.max(counts))

        if best_k is None or best_k not in centroids:
            saved = 0
            for i in idx_t:
                src = Path(img_paths[int(i)]).resolve()
                if not src.exists():
                    continue
                if str(src) in used_paths:
                    continue

                dst = class_dir / f"rank{saved+1:02d}_fallback_{src.name}"
                if dst.exists():
                    dst = class_dir / f"rank{saved+1:02d}_fallback_{src.stem}_{int(i)}{src.suffix}"

                shutil.copy2(src, dst)
                used_paths.add(str(src))
                saved += 1
                if saved >= topk:
                    break
            continue

        mu = centroids[best_k]

        idx = idx_t[(y_pred[idx_t] == best_k)]
        if len(idx) == 0:
            idx = idx_t_non_noise
            if len(idx) == 0:
                continue

            dlist = []
            for i in idx:
                k_i = int(y_pred[int(i)])
                mu_i = centroids.get(k_i, None)
                if mu_i is None:
                    continue
                d = float(np.sum((X[int(i)] - mu_i) ** 2))
                dlist.append((d, int(i)))
            dlist.sort(key=lambda x: x[0])

            saved = 0
            for d, i in dlist:
                src = Path(img_paths[int(i)]).resolve()
                if not src.exists():
                    continue
                if str(src) in used_paths:
                    continue
                dst = class_dir / f"rank{saved+1:02d}_d{d:.4f}_{src.name}"
                if dst.exists():
                    dst = class_dir / f"rank{saved+1:02d}_d{d:.4f}_{src.stem}_{int(i)}{src.suffix}"
                shutil.copy2(src, dst)
                used_paths.add(str(src))
                saved += 1
                if saved >= topk:
                    break
            continue

        d = np.sum((X[idx] - mu) ** 2, axis=1)
        order = np.argsort(d)

        saved = 0
        for j in order:
            i = int(idx[int(j)])
            src = Path(img_paths[i]).resolve()
            if not src.exists():
                continue
            if str(src) in used_paths:
                continue

            dst = class_dir / f"rank{saved+1:02d}_k{best_k}_d{float(d[int(j)]):.4f}_{src.name}"
            if dst.exists():
                dst = class_dir / f"rank{saved+1:02d}_k{best_k}_d{float(d[int(j)]):.4f}_{src.stem}_{i}{src.suffix}"

            shutil.copy2(src, dst)
            used_paths.add(str(src))
            saved += 1
            if saved >= topk:
                break



def main():
    splits_dir = Path(SPLITS_DIR)
    classes, class_to_label = read_labels_json(splits_dir)
    n_classes = len(classes)

    out_dir = Path("outputs") / MODEL_NAME / FEATURE_NAME
    ensure_dir(out_dir)

    extractor = pick_extractor(FEATURE_NAME)

    tr_paths, y_tr = load_split_folder(splits_dir, "train", classes, class_to_label)
    va_paths, y_va = load_split_folder(splits_dir, "val", classes, class_to_label)
    te_paths, y_te = load_split_folder(splits_dir, "test", classes, class_to_label)

    # extract features
    X_tr = np.asarray(extractor(tr_paths), dtype=np.float32)
    X_va = np.asarray(extractor(va_paths), dtype=np.float32)
    X_te = np.asarray(extractor(te_paths), dtype=np.float32)

    # safe PCA grid
    max_dim = int(min(X_tr.shape[0], X_tr.shape[1]))
    GRID_PCA_SAFE = [int(p) for p in GRID_PCA if int(p) <= max_dim]
    if len(GRID_PCA_SAFE) == 0:
        GRID_PCA_SAFE = [max_dim]

    print(f"[INFO] X_tr shape = {X_tr.shape}; max PCA dim = {max_dim}")
    print(f"[INFO] GRID_PCA_SAFE = {GRID_PCA_SAFE}")

    all_val_results = []
    best = None
    best_ok = None  # best satisfying MIN_CLUSTERS_REQUIRED

    for pca_dim in GRID_PCA_SAFE:
        # Fit scaler + PCA on TRAIN only
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr).astype(np.float32)

        pca = PCA(n_components=int(pca_dim), random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)

        Xva_p = pca.transform(scaler.transform(X_va)).astype(np.float32)
        Xte_p = pca.transform(scaler.transform(X_te)).astype(np.float32)

        for metric in GRID_METRIC:
            for eps in GRID_EPS:
                for ms in GRID_MIN_SAMPLES:
                    pred_va = DBSCAN(
                        eps=float(eps),
                        min_samples=int(ms),
                        metric=metric,
                        n_jobs=-1
                    ).fit_predict(Xva_p)

                    met = clustering_metrics(y_va, pred_va)
                    n_clusters, n_noise = count_clusters_and_noise(pred_va)

                    sel_score = selection_score(met["NMI"], n_clusters, n_noise)

                    res = {
                        "model": "DBSCAN",
                        "feature": FEATURE_NAME,
                        "params": {"pca_dim": int(pca_dim), "metric": metric, "eps": float(eps), "min_samples": int(ms)},
                        "val": met,
                        "val_clusters_found": int(n_clusters),
                        "val_noise_points": int(n_noise),
                        "val_selection_score": float(sel_score),
                    }
                    all_val_results.append(res)

                    # track best overall by selection score
                    if best is None or sel_score > best["score"]:
                        best = {"score": sel_score, "res": res, "Xte_p": Xte_p}

                    # track best that satisfies MIN_CLUSTERS_REQUIRED
                    if n_clusters >= MIN_CLUSTERS_REQUIRED:
                        if best_ok is None or sel_score > best_ok["score"]:
                            best_ok = {"score": sel_score, "res": res, "Xte_p": Xte_p}

    chosen = best_ok if best_ok is not None else best
    bp = chosen["res"]["params"]

    # plots
    plot_best_metric_vs_pca(out_dir, all_val_results, "NMI")
    plot_best_metric_vs_pca(out_dir, all_val_results, "HungAcc")
    plot_best_metric_vs_pca(out_dir, all_val_results, "ARI")
    plot_best_fairscore_vs_pca(out_dir, all_val_results)

    # final on TEST
    pred_te = DBSCAN(
        eps=float(bp["eps"]),
        min_samples=int(bp["min_samples"]),
        metric=bp["metric"],
        n_jobs=-1
    ).fit_predict(chosen["Xte_p"])
    test_met = clustering_metrics(y_te, pred_te)
    te_clusters, te_noise = count_clusters_and_noise(pred_te)

    # representatives
    save_representatives_dbscan(
        out_dir=out_dir,
        img_paths=te_paths,
        X_embed=chosen["Xte_p"],  # embedding-ul PCA pentru test
        y_true=y_te,
        y_pred=pred_te,  # labels DBSCAN pe test
        n_classes=n_classes,
        topk=TOPK_REP,
    )

    out = {
        "model": "DBSCAN",
        "feature": FEATURE_NAME,
        "classes": classes,
        "selection": {
            "use_fair_score": USE_FAIR_SCORE,
            "target_clusters": TARGET_CLUSTERS,
            "min_clusters_required": MIN_CLUSTERS_REQUIRED,
            "lambda_clusters": LAMBDA_CLUSTERS,
            "lambda_noise": LAMBDA_NOISE,
        },
        "best_by": "selection_score_on_val" if USE_FAIR_SCORE else "NMI_on_val",
        "best_params": bp,
        "val_best": chosen["res"]["val"],
        "val_clusters_found": int(chosen["res"]["val_clusters_found"]),
        "val_noise_points": int(chosen["res"]["val_noise_points"]),
        "val_selection_score": float(chosen["res"]["val_selection_score"]),
        "test_metrics": test_met,
        "test_clusters_found": int(te_clusters),
        "test_noise_points": int(te_noise),
        "test_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(pred_te, return_counts=True))},
        "all_val_results": all_val_results,
        "artifacts": {
            "plots": [
                "val_best_nmi_vs_pca_dim.png",
                "val_best_hungacc_vs_pca_dim.png",
                "val_best_ari_vs_pca_dim.png",
                "val_best_selection_score_vs_pca_dim.png" if USE_FAIR_SCORE else None,
            ],
            "representatives_dir": "representatives/",
        },
    }

    out["artifacts"]["plots"] = [p for p in out["artifacts"]["plots"] if p is not None]

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_dir / "results.json")
    print("VAL best:", out["val_best"])
    print("VAL clusters/noise:", out["val_clusters_found"], out["val_noise_points"])
    print("TEST:", out["test_metrics"])
    print("TEST clusters/noise:", out["test_clusters_found"], out["test_noise_points"])
    print("Best params:", out["best_params"])
    print("VAL selection score:", out["val_selection_score"])


if __name__ == "__main__":
    main()

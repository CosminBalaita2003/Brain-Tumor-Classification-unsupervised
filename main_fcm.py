# main_fcm_pca.py
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from feature_extraction_hog import extract_features as extract_hog
from feature_extraction_handcrafted import extract_features as extract_hand



SPLITS_DIR = "splits_brain_tumor"
MODEL_NAME = "fcm"
FEATURE_NAME = "handcrafted"            # "hog" or "handcrafted"
RANDOM_STATE = 42

# PCA grid
GRID_PCA = [10, 20, 30, 50]

# FCM grid (VAL)
GRID_M = [1.2, 1.4, 1.6, 1.8, 2.0]
GRID_METRIC = ["euclidean", "cosine"]

TOPK_REP = 12



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

    for c in pred_labels:
        c = int(c)
        if c not in mapping:
            mapping[c] = -1

    acc = correct / len(y_true)
    return float(acc), mapping


def clustering_metrics(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    hacc, mapping = hungarian_map_accuracy(y_true, y_pred)
    return {"ARI": float(ari), "NMI": float(nmi), "HungAcc": float(hacc), "mapping": mapping}


def pairwise_sqeuclid(X, V):
    X2 = np.sum(X * X, axis=1, keepdims=True)
    V2 = np.sum(V * V, axis=1, keepdims=True).T
    XV = X @ V.T
    D = X2 + V2 - 2.0 * XV
    return np.maximum(D, 1e-12)


def pairwise_cosine_dist(X, V):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Vn.T
    D = 1.0 - sim
    return np.maximum(D, 1e-12)


def fcm_fit_predict(X, c, m=2.0, metric="euclidean", max_iter=400, tol=1e-6, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    U = rng.random((n, c))
    U = U / (U.sum(axis=1, keepdims=True) + 1e-12)

    def dist_fn(A, B):
        if metric == "euclidean":
            return pairwise_sqeuclid(A, B)
        if metric == "cosine":
            return pairwise_cosine_dist(A, B)
        raise ValueError(f"Unknown metric: {metric}")

    prev_obj = None
    for _ in range(max_iter):
        Um = U ** m
        denom = Um.sum(axis=0, keepdims=True).T
        V = (Um.T @ X) / (denom + 1e-12)

        D = dist_fn(X, V)
        obj = float(np.sum(Um * D))

        power = 1.0 / (m - 1.0)
        ratio = (D[:, :, None] / (D[:, None, :] + 1e-12)) ** power
        U_new = 1.0 / (np.sum(ratio, axis=2) + 1e-12)
        U_new = U_new / (U_new.sum(axis=1, keepdims=True) + 1e-12)

        if prev_obj is not None and abs(prev_obj - obj) < tol:
            U = U_new
            prev_obj = obj
            break

        U = U_new
        prev_obj = obj

    labels = np.argmax(U, axis=1).astype(int)
    return labels, U.astype(np.float32), V.astype(np.float32), float(prev_obj)


def plot_best_vs_pca(out_dir: Path, all_val_results, metric_name: str):
    xs, ys = [], []
    for p in GRID_PCA:
        subset = [r for r in all_val_results if int(r["params"]["pca_dim"]) == int(p)]
        if not subset:
            continue
        best = max(subset, key=lambda r: float(r["val"][metric_name]))
        xs.append(p)
        ys.append(float(best["val"][metric_name]))

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("PCA dim")
    plt.ylabel(f"Best VAL {metric_name}")
    plt.title(f"FCM: Best VAL {metric_name} vs PCA dim")
    plt.tight_layout()
    plt.savefig(out_dir / f"val_best_{metric_name.lower()}_vs_pca_dim.png", dpi=150)
    plt.close()


def plot_val_curves_for_best_pca(out_dir: Path, rows_for_best_pca):
    """
    rows_for_best_pca: all results for a fixed pca_dim (varies m, metric)
    Saves val_*_vs_m.png
    """
    for met_name in ["NMI", "HungAcc", "ARI"]:
        plt.figure()
        for metric in GRID_METRIC:
            xs, ys = [], []
            for m in GRID_M:
                hit = None
                for r in rows_for_best_pca:
                    p = r["params"]
                    if float(p["m"]) == float(m) and p["metric"] == metric:
                        hit = r
                        break
                if hit:
                    xs.append(m)
                    ys.append(hit["val"][met_name])
            if xs:
                plt.plot(xs, ys, marker="o", label=metric)

        plt.xlabel("m (fuzziness)")
        plt.ylabel(f"VAL {met_name}")
        plt.title(f"FCM (best PCA dim): VAL {met_name} vs m")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"val_{met_name.lower()}_vs_m.png", dpi=150)
        plt.close()


def save_representatives_by_true_class(out_dir: Path, img_paths, y_true, U, mapping_cluster_to_true, n_classes, topk=12):

    reps_root = out_dir / "representatives"
    ensure_dir(reps_root)

    inv = {}
    for k, t in mapping_cluster_to_true.items():
        inv[int(t)] = int(k)

    y_true = np.asarray(y_true, dtype=int)
    U = np.asarray(U, dtype=float)

    for t in range(n_classes):
        class_dir = reps_root / f"class_{t}"
        ensure_dir(class_dir)

        if t not in inv:
            continue
        k = inv[t]

        idx = np.where(y_true == t)[0]
        if len(idx) == 0:
            continue

        scores = U[idx, k]
        order = np.argsort(-scores)[:topk]
        chosen = idx[order]

        for rank, i in enumerate(chosen, start=1):
            src = Path(img_paths[i])
            if not src.exists():
                continue
            dst = class_dir / f"rank{rank:02d}_u{scores[order[rank-1]]:.4f}_{src.name}"
            shutil.copy2(src, dst)



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

    # Extract once
    X_tr = np.asarray(extractor(tr_paths), dtype=np.float32)
    X_va = np.asarray(extractor(va_paths), dtype=np.float32)
    X_te = np.asarray(extractor(te_paths), dtype=np.float32)

    all_val_results = []
    best = None

    for pca_dim in GRID_PCA:
        # scaler + PCA fit on TRAIN only
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr).astype(np.float32)

        pca = PCA(n_components=int(pca_dim), random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_s).astype(np.float32)

        Xva_p = pca.transform(scaler.transform(X_va)).astype(np.float32)
        Xte_p = pca.transform(scaler.transform(X_te)).astype(np.float32)

        # grid on VAL for this pca_dim
        for m in GRID_M:
            for metric in GRID_METRIC:
                pred_va, U_va, _, obj_va = fcm_fit_predict(
                    Xva_p, c=n_classes, m=m, metric=metric,
                    max_iter=450, tol=1e-6, random_state=RANDOM_STATE
                )
                met = clustering_metrics(y_va, pred_va)

                res = {
                    "model": "FCM",
                    "feature": FEATURE_NAME,
                    "params": {"pca_dim": int(pca_dim), "c": n_classes, "m": float(m), "metric": metric},
                    "val": met,
                    "val_obj": float(obj_va),
                    "val_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(pred_va, return_counts=True))},
                }
                all_val_results.append(res)

                score = met["NMI"]
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "res": res,
                        "scaler": scaler,
                        "pca": pca,
                        "Xte_p": Xte_p,
                    }

    plot_best_vs_pca(out_dir, all_val_results, "NMI")
    plot_best_vs_pca(out_dir, all_val_results, "HungAcc")
    plot_best_vs_pca(out_dir, all_val_results, "ARI")

    best_pca_dim = int(best["res"]["params"]["pca_dim"])
    rows_best_pca = [r for r in all_val_results if int(r["params"]["pca_dim"]) == best_pca_dim]
    plot_val_curves_for_best_pca(out_dir, rows_best_pca)

    bp = best["res"]["params"]
    pred_te, U_te, _, obj_te = fcm_fit_predict(
        best["Xte_p"], c=n_classes, m=bp["m"], metric=bp["metric"],
        max_iter=700, tol=1e-6, random_state=RANDOM_STATE
    )
    test_met = clustering_metrics(y_te, pred_te)

    save_representatives_by_true_class(
        out_dir=out_dir,
        img_paths=te_paths,
        y_true=y_te,
        U=U_te,
        mapping_cluster_to_true=test_met["mapping"],
        n_classes=n_classes,
        topk=TOPK_REP,
    )

    out = {
        "model": "FCM",
        "feature": FEATURE_NAME,
        "classes": classes,
        "best_by": "NMI_on_val",
        "best_params": {
            "pca_dim": int(bp["pca_dim"]),
            "c": int(bp["c"]),
            "m": float(bp["m"]),
            "metric": bp["metric"],
        },
        "val_best": best["res"]["val"],
        "test_metrics": test_met,
        "test_obj": float(obj_te),
        "test_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(pred_te, return_counts=True))},
        "all_val_results": all_val_results,
        "artifacts": {
            "plots": [
                "val_best_nmi_vs_pca_dim.png",
                "val_best_hungacc_vs_pca_dim.png",
                "val_best_ari_vs_pca_dim.png",
                "val_nmi_vs_m.png",
                "val_hungacc_vs_m.png",
                "val_ari_vs_m.png",
            ],
            "representatives_dir": "representatives/",
        },
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", out_dir / "results.json")
    print("VAL best:", out["val_best"])
    print("TEST:", out["test_metrics"])
    print("Best params:", out["best_params"])


if __name__ == "__main__":
    main()

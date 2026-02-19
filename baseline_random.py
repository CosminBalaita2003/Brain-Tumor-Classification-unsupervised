import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


SPLITS_DIR = "splits_brain_tumor"
OUT_DIR = Path("outputs") / "baselines" / "random"
SEED = 42


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


def load_test_labels(splits_dir: Path, classes, class_to_label):
    y = []
    for cls in classes:
        cls_dir = splits_dir / "test" / cls
        if not cls_dir.exists():
            continue
        imgs = list_images(cls_dir)
        y.extend([class_to_label[cls]] * len(imgs))
    return np.asarray(y, dtype=int)


def hungarian_map_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    C = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)
    pred_to_row = {c: i for i, c in enumerate(pred_labels)}
    true_to_col = {c: j for j, c in enumerate(true_labels)}

    for t, p in zip(y_true, y_pred):
        C[pred_to_row[p], true_to_col[t]] += 1

    row_ind, col_ind = linear_sum_assignment(-C)

    correct = 0
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[int(pred_labels[r])] = int(true_labels[c])
        correct += C[r, c]

    return float(correct / len(y_true)), mapping


def main():
    splits_dir = Path(SPLITS_DIR)
    classes, class_to_label = read_labels_json(splits_dir)
    n_classes = len(classes)

    ensure_dir(OUT_DIR)

    y_true = load_test_labels(splits_dir, classes, class_to_label)

    rng = np.random.default_rng(SEED)
    y_pred = rng.integers(0, n_classes, size=len(y_true))

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    hacc, mapping = hungarian_map_accuracy(y_true, y_pred)

    out = {
        "baseline": "random",
        "seed": SEED,
        "classes": classes,
        "test_metrics": {"ARI": float(ari), "NMI": float(nmi), "HungAcc": float(hacc), "mapping": mapping},
    }

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUT_DIR / "results.json")
    print(out["test_metrics"])


if __name__ == "__main__":
    main()

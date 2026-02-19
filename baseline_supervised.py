import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

from feature_extraction_hog import extract_features as extract_hog
from feature_extraction_handcrafted import extract_features as extract_hand


SPLITS_DIR = "splits_brain_tumor"
FEATURE_NAME = "hog"  #"handcrafted"
OUT_DIR = Path("outputs") / "baselines" / "supervised" / FEATURE_NAME

GRID_C = [0.1, 1.0, 10.0]
RANDOM_STATE = 42


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


def save_confusion_matrix_png(out_path: Path, cm: np.ndarray):
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix (TEST)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    splits_dir = Path(SPLITS_DIR)
    classes, class_to_label = read_labels_json(splits_dir)

    ensure_dir(OUT_DIR)

    extractor = pick_extractor(FEATURE_NAME)

    tr_paths, y_tr = load_split_folder(splits_dir, "train", classes, class_to_label)
    va_paths, y_va = load_split_folder(splits_dir, "val", classes, class_to_label)
    te_paths, y_te = load_split_folder(splits_dir, "test", classes, class_to_label)

    X_tr = np.asarray(extractor(tr_paths), dtype=np.float32)
    X_va = np.asarray(extractor(va_paths), dtype=np.float32)
    X_te = np.asarray(extractor(te_paths), dtype=np.float32)

    scaler = StandardScaler()
    Xtr_n = scaler.fit_transform(X_tr).astype(np.float32)
    Xva_n = scaler.transform(X_va).astype(np.float32)
    Xte_n = scaler.transform(X_te).astype(np.float32)

    best = None
    curve = []
    for C in GRID_C:
        clf = LinearSVC(C=float(C), random_state=RANDOM_STATE)
        clf.fit(Xtr_n, y_tr)
        pred_va = clf.predict(Xva_n)
        score = f1_score(y_va, pred_va, average="macro")
        curve.append({"C": float(C), "val_macro_f1": float(score)})
        if best is None or score > best["score"]:
            best = {"score": score, "C": float(C), "clf": clf}

    # plot curve
    plt.figure()
    plt.plot([r["C"] for r in curve], [r["val_macro_f1"] for r in curve], marker="o")
    plt.xlabel("C")
    plt.ylabel("VAL Macro-F1")
    plt.title("LinearSVC: VAL Macro-F1 vs C")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "val_macro_f1_vs_C.png", dpi=150)
    plt.close()

    pred_te = best["clf"].predict(Xte_n)
    cm = confusion_matrix(y_te, pred_te)
    save_confusion_matrix_png(OUT_DIR / "confusion_matrix_test.png", cm)

    out = {
        "baseline": "supervised_linear_svc",
        "feature": FEATURE_NAME,
        "classes": classes,
        "best_by": "macro_f1_on_val",
        "best_C": best["C"],
        "val_curve": curve,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_te, pred_te)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred_te)),
            "macro_f1": float(f1_score(y_te, pred_te, average="macro")),
        },
        "artifacts": {"plots": ["val_macro_f1_vs_C.png", "confusion_matrix_test.png"]},
    }

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUT_DIR / "results.json")
    print("TEST:", out["test_metrics"])


if __name__ == "__main__":
    main()

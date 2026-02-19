from pathlib import Path
import random
import shutil
import json
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def copy_files(files: List[Path], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def split_train_val_test(
    dataset_root: str,
    out_dir: str = "splits_brain_tumor",
    val_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)

    dataset_root = Path(dataset_root)
    train_root = dataset_root / "Training"
    test_root  = dataset_root / "Testing"
    out_dir    = Path(out_dir)

    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Expected {train_root} and {test_root}. "
            "Point dataset_root to the folder that contains Training/ and Testing/."
        )

    # Prepare output dirs
    ensure_empty_dir(out_dir)
    (out_dir / "train").mkdir()
    (out_dir / "val").mkdir()
    (out_dir / "test").mkdir()

    classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {train_root}")

    class_to_label = {cls: i for i, cls in enumerate(classes)}
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_label": class_to_label}, f, indent=2)

    # Split Training -> train/val
    for cls in classes:
        cls_dir = train_root / cls
        imgs = list_images(cls_dir)
        if len(imgs) == 0:
            raise RuntimeError(f"No images found in {cls_dir}")

        random.shuffle(imgs)
        n_val = int(round(len(imgs) * val_ratio))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        copy_files(train_imgs, out_dir / "train" / cls)
        copy_files(val_imgs,   out_dir / "val"   / cls)

    # Copy Testing -> test
    test_classes = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    for cls in test_classes:
        imgs = list_images(test_root / cls)
        if len(imgs) == 0:
            continue
        copy_files(imgs, out_dir / "test" / cls)

    def count_split(split: str):
        return sum(1 for _ in (out_dir / split).rglob("*") if _.is_file())

    print("Done.")
    print("Classes:", classes)
    print("Saved mapping to:", out_dir / "labels.json")
    print("Counts:",
          "train =", count_split("train"),
          "val   =", count_split("val"),
          "test  =", count_split("test"))


if __name__ == "__main__":
    split_train_val_test(
        dataset_root="Dataset",
        out_dir="splits_brain_tumor",
        val_ratio=0.15,
        seed=42,
    )

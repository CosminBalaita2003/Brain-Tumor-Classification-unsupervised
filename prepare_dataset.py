"""Create reproducible binary splits without changing the original archive."""

from hashlib import sha256
import json
from pathlib import Path
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {"not_tumor": 0, "tumor": 1}
SOURCE_TO_BINARY = {
    "no_tumor": "not_tumor",
    "glioma_tumor": "tumor",
    "meningioma_tumor": "tumor",
    "pituitary_tumor": "tumor",
}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> list[tuple[Path, str, str]]:
    records = []
    for source_class, binary_class in SOURCE_TO_BINARY.items():
        for path in list_images(root / source_class):
            records.append((path, source_class, binary_class))
    return records


def copy_records(records, destination: Path) -> None:
    for source, source_class, binary_class in records:
        target_dir = destination / binary_class
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source_class}__{source.name}"
        shutil.copy2(source, target)


def prepare_dataset(
    dataset_root: str = "archive",
    output_dir: str = "splits_brain_tumor",
    validation_ratio: float = 0.20,
    seed: int = 42,
) -> None:
    source_root = Path(dataset_root)
    train_root = source_root / "Training"
    test_root = source_root / "Testing"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(
            f"Expected {train_root} and {test_root}; archive was not modified."
        )

    test_records = collect(test_root)
    test_hashes = {file_hash(record[0]) for record in test_records}
    unique_training = []
    seen_training_hashes = set()
    excluded = 0
    for record in collect(train_root):
        digest = file_hash(record[0])
        if digest in test_hashes or digest in seen_training_hashes:
            excluded += 1
            continue
        seen_training_hashes.add(digest)
        unique_training.append(record)

    rng = random.Random(seed)
    training_records = []
    validation_records = []
    for binary_class in CLASS_TO_LABEL:
        group = [r for r in unique_training if r[2] == binary_class]
        rng.shuffle(group)
        validation_size = round(len(group) * validation_ratio)
        validation_records.extend(group[:validation_size])
        training_records.extend(group[validation_size:])

    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    for split in ("train", "val", "test"):
        for class_name in CLASS_TO_LABEL:
            (destination / split / class_name).mkdir(parents=True, exist_ok=True)

    copy_records(training_records, destination / "train")
    copy_records(validation_records, destination / "val")
    copy_records(test_records, destination / "test")

    metadata = {
        "classes": list(CLASS_TO_LABEL),
        "class_to_label": CLASS_TO_LABEL,
        "source_to_binary": SOURCE_TO_BINARY,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "duplicates_excluded_from_training": excluded,
    }
    (destination / "labels.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Binary dataset prepared from archive (archive was not modified).")
    for split, records in (
        ("train", training_records),
        ("val", validation_records),
        ("test", test_records),
    ):
        counts = {
            class_name: sum(r[2] == class_name for r in records)
            for class_name in CLASS_TO_LABEL
        }
        print(f"{split}: {len(records)} images {counts}")
    print(f"Excluded duplicate/leaking training images: {excluded}")


if __name__ == "__main__":
    prepare_dataset()

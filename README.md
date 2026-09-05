# Binary Brain MRI Clustering

This project clusters MRI images into two groups: `not_tumor` (label 0) and
`tumor` (label 1). Glioma, meningioma, and pituitary images are combined into
the tumor class. Labels are never passed to KMeans or DBSCAN; they are used
only after clustering for evaluation and interpretation.

## Grading coverage

- Two clustering models: KMeans and DBSCAN.
- Two genuinely different representations: HOG shape features and handcrafted
  intensity/texture features (statistics, histogram, LBP, and GLCM).
- Validation grid searches with hyperparameter plots. DBSCAN also compares
  Euclidean and cosine distances.
- Random-chance and supervised LinearSVC baselines.
- ARI, NMI, silhouette score, mapped accuracy, confusion matrices, and
  representative images for qualitative cluster interpretation.
- Scaler and PCA are fitted on training features only; parameters are selected
  on validation; test labels are reserved for final evaluation.

## Run

From this directory:

```bash
python prepare_dataset.py
python main_kmeans.py
python main_dbscan.py
python baseline_random.py
python baseline_supervised.py
```

`prepare_dataset.py` reads `archive/` and recreates only
`splits_brain_tumor/`; it does not modify the original images. It removes exact
duplicates that could leak from the original training partition into the test
partition.

Results are stored under `outputs/<model>/<feature>/`. Each clustering run
writes `results.json`, validation/test confusion matrices, a grid-search plot,
and representative test images grouped by discovered cluster.

Each model script runs both feature configurations sequentially: HOG first,
then handcrafted features.

## Submission note

The grading PDF requires all `.py` files in one folder named
`P2_{family_name}_{first_name}_{group}`, the report PDF in the corresponding
`_doc` subfolder, and a ZIP containing code and documentation but no dataset.

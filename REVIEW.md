# Week 1 Code Review: K-Means Implementation

## What is working well
- Pipeline is split into small functions (load_data, preprocess,
  validate, cluster, run_pipeline), so each part is easy to test.
- Features are scaled with StandardScaler before clustering.
- K-Means++ initialization and a fixed random_state give stable,
  repeatable results.
- Edge cases are handled: empty data, missing columns, invalid
  n_clusters, NaN or infinite values.
- End-to-end tests cover the full workflow.

## Issues found
- The dataset has only 20 sample rows, so results are not fully
  representative. The full Mall Customers dataset (200 rows) should be used.
- K is fixed at 3. It is not chosen using Elbow or Silhouette methods.
- Notebooks repeat some code. The functions should move into a
  reusable Python module (e.g., segmentation.py).
- No evaluation metrics are reported yet.
- Visualization is basic (one scatter plot).

## Improvements planned
- Use the full dataset.
- Choose K using the Elbow method and Silhouette score.
- Add better plots and cluster profiles.
- Move shared code into a .py module.

- ## Week 2 Review: Issues Fixed and Remaining

### Fixed in Week 2
- Shared code moved into a reusable module (`segmentation.py`), so
  notebooks no longer repeat the same functions.
- Evaluation metrics added: Silhouette score and Calinski-Harabasz index.
- Visualization improved: PCA 2D plot with centroids and cluster
  profile bar chart.
- Interactive dashboard added to explore clusters with different K values.
- Hyperparameter tuning added (K, init, n_init, max_iter) using
  silhouette score.
- Unit tests added for evaluation metrics and visualization.
- Performance testing added for different data sizes.
- Refactored module tested in `refactor_test.ipynb`.

### Still pending
- The dataset still has only 20 sample rows. The full Mall Customers
  dataset (200 rows) should be used for final results.
- Notebooks still contain some older copies of the functions. They can
  be updated to import from `segmentation.py`.
- Final README update with metrics, plots and cluster descriptions.
  

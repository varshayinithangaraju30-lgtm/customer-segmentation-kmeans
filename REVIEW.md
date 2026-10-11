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
  

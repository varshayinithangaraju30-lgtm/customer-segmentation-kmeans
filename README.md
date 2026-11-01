# Customer Segmentation using K-Means

## What is K-Means?
K-Means is an unsupervised machine learning algorithm that divides data
into K groups (clusters) based on similarity. Points that are close to
each other end up in the same cluster.

## How the Algorithm Works
1. Choose the number of clusters, K.
2. Initialize K centroids (using K-Means++ for better spread).
3. Assign each data point to its nearest centroid.
4. Recalculate each centroid as the mean of its assigned points.
5. Repeat steps 3 and 4 until convergence.

## Centroid Initialization
Random initialization can lead to poor clusters. K-Means++ picks the
first centroid randomly and chooses the next ones far from existing
centroids, giving stable and better results.

## Convergence Criteria
The algorithm stops when:
- Centroids no longer move (or move less than a tolerance value)
- Cluster assignments stop changing
- The maximum number of iterations is reached

## Choosing K
- **Elbow Method:** Plot inertia vs K and pick the "elbow" point.
- **Silhouette Score:** Higher score means better-separated clusters.

## Use in Customer Segmentation
Customers are grouped by features like Annual Income and Spending Score.
Each cluster represents a customer type (e.g., high income + high
spending = best customers), helping businesses target offers better.

## Limitations
- K must be chosen in advance.
- Sensitive to outliers.
- Works best with spherical clusters.
- Features must be scaled.
- Not suitable for categorical data.

- ## Data Source
Mall Customers dataset (Kaggle). Features: CustomerID, Gender, Age,
Annual Income (k$), Spending Score (1-100).

## Preprocessing Steps
1. Removed duplicate rows and missing values.
2. Encoded Gender (Male=0, Female=1).
3. Selected features: Gender, Age, Annual Income, Spending Score.
4. Normalized features using StandardScaler (mean 0, std 1).
5. Validated data: no NaN, no infinite values, enough rows for clustering.

6. ## Implementation Details

The project is implemented in Python using pandas, NumPy, scikit-learn
and matplotlib.

### Pipeline Structure
- `load_data()` loads the dataset and checks it is not empty.
- `preprocess()` removes duplicates and missing values, then scales
  features using StandardScaler.
- `validate()` checks for NaN or infinite values and a valid number of clusters.
- `cluster()` runs K-Means with K-Means++ initialization.
- `run_pipeline()` connects all the steps and returns the segmented data.

### Notebooks
- `segmentation.ipynb`: first basic K-Means implementation
- `data_pipeline.ipynb`: data preparation and validation
- `core_segmentation.ipynb`: core feature with edge cases and tests
- `integration.ipynb`: complete end-to-end pipeline with tests

### Testing
End-to-end tests check that the pipeline adds a Cluster column, gives the
expected number of clusters, produces the same result on every run, and
raises errors for invalid inputs.

### Results
Using K=3, customers are grouped into three segments:
- Young customers with high spending scores
- Older customers with low spending scores
- Younger customers with medium spending scores

- ## Advanced Features (Week 3)

### Hierarchical Clustering and DBSCAN
- `advanced_clustering.ipynb` adds Hierarchical (agglomerative) clustering
  and DBSCAN alongside K-Means.
- Hierarchical clustering does not need K to be chosen in advance.
- DBSCAN finds clusters of any shape and marks outliers as noise (-1).
- All three methods are compared using the silhouette score and
  PCA 2D plots.

### Interactive Dashboard
- `dashboard_ui.ipynb` provides an interactive dashboard with a K slider
  and X/Y axis dropdowns.
- Each cluster gets a readable name (e.g., "Young High Spenders") and a
  summary card showing customer count and average age, income and spending.
- Hovering over a point shows customer details.

## Optimization Techniques

### Parallel Processing and Caching
- `performance_optimization.ipynb` runs K-Means for different K values in
  parallel using `joblib.Parallel` on all CPU cores.
- Results are cached with `joblib.Memory`, so a repeated run returns
  almost instantly.
- Tests check that parallel and cached results match the sequential run.

### Hyperparameter Tuning
- `hyperparameter_tuning.ipynb` searches over K, init, n_init and max_iter
  and picks the best combination by silhouette score.

## Code Quality, Logging and Monitoring

### Code Quality
- `segmentation.py` is a reusable module with type hints, docstrings,
  logging and clear error messages.
- Inputs are validated (missing columns, non-numeric data, invalid K).

### Logging and Monitoring
- `monitored_pipeline.py` logs every workflow step to `segmentation.log`.
- Metrics (rows used, cluster sizes, inertia, silhouette score, run time)
  are saved to `metrics.json`.
- Warnings are logged when the silhouette score is low, a cluster is very
  small, or the workflow is slow.

## Testing
- Unit tests cover evaluation metrics, visualization, the segmentation
  module, parallel processing and the monitored pipeline.
- Performance tests check run time and memory for up to 20,000 rows.

## Notebooks Overview
| Notebook | Purpose |
|----------|---------|
| `evaluation_metrics.ipynb` | Silhouette and Calinski-Harabasz scores |
| `visualization.ipynb` | PCA plots and cluster profiles |
| `dashboard_ui.ipynb` | Interactive dashboard |
| `hyperparameter_tuning.ipynb` | Grid search for K-Means settings |
| `advanced_clustering.ipynb` | Hierarchical clustering and DBSCAN |
| `performance_optimization.ipynb` | Parallel processing and caching |
| `code_quality.ipynb` | Improved module with logging and validation |
| `logging_monitoring.ipynb` | Logging and monitoring |

## Limitations and Next Steps
- The dataset in this repo is a 20-row sample. The full Mall Customers
  dataset (200 rows) should be used for final results.
- User testing feedback will be added after responses are collected.

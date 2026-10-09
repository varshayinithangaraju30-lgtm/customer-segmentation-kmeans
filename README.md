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

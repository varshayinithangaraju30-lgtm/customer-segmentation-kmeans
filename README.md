# 🛍️ Customer Segmentation using K-Means

An interactive dashboard that groups mall customers into segments based on **Age**, **Annual Income** and **Spending Score**, so a business can understand its customers and target each group differently.

## 🔗 Links

- **Live Dashboard:** https://customer-segmentation-kmeans-bobgzp9zc646x7glsbnmzt.streamlit.app
- **Demo Video:** https://youtu.be/C9b715CfiEs

## 📌 Problem Statement

Businesses cannot treat all customers the same. This project uses unsupervised machine learning (K-Means clustering) to automatically discover customer groups such as high-income high-spenders, low-income low-spenders, and so on.

## 📊 Dataset

- **File:** `Mall_Customers.csv`
- **Features used:** Age, Annual Income (k$), Spending Score (1-100)

## 🧠 Approach

1. **Data cleaning:** removed duplicate rows and rows with missing values in the selected features.
2. **Scaling:** applied `StandardScaler` so that all features contribute equally to the distance calculation.
3. **Clustering:** trained a `KMeans` model (`n_init=10`, `random_state=42`).
4. **Choosing K:** compared K = 2 to 10 using the **Elbow method** (inertia) and the **Silhouette score**.
5. **Visualisation:** plotted clusters using Plotly scatter plots, a 2D PCA projection, and a 3D plot.
6. **Comparison:** also experimented with Hierarchical clustering and DBSCAN in the notebooks, and compared them using the silhouette score.

## ✨ Dashboard Features

- Slider to change the number of clusters (K)
- Choose which features to use for clustering
- Metrics: number of customers, clusters, silhouette score
- **Cluster plot** tab: Income vs Spending Score scatter, PCA view, 3D view
- **Cluster summary** tab: average values and size of each cluster
- **Best K** tab: elbow curve and silhouette curve
- **Data** tab: view the clustered data and download it as CSV

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Streamlit, Google Colab, GitHub

## 📁 Project Structure

```
customer-segmentation-kmeans/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── Mall_Customers.csv          # Dataset
├── core_segmentation.ipynb     # Core K-Means segmentation
├── data_pipeline.ipynb         # Data pipeline and validation
├── hyperparameter_tuning.ipynb # Tuning K
├── evaluation_metrics.ipynb    # Silhouette and Calinski-Harabasz
├── dashboard_v2.ipynb          # Interactive Plotly dashboard
├── advanced_clustering.ipynb   # Hierarchical and DBSCAN comparison
└── README.md
```

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/varshayinithangaraju30-lgtm/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🌐 Deployment

The dashboard is deployed on **Streamlit Community Cloud** directly from this GitHub repository.

## 📈 Results

With K = 5 the silhouette score is about **0.37** on the dataset in this repository. The dashboard lets you change K and see how the segments and the score change.

## 👩‍💻 Author

**Varshayini Thangaraju**
GitHub: [varshayinithangaraju30-lgtm](https://github.com/varshayinithangaraju30-lgtm)

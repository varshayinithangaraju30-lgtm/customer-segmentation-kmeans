import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

URL = (
    "https://raw.githubusercontent.com/varshayinithangaraju30-lgtm/"
    "customer-segmentation-kmeans/main/Mall_Customers.csv"
)
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


@st.cache_data
def load_data():
    return pd.read_csv(URL).drop_duplicates().dropna(subset=FEATURES)


df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of clusters (K)", 2, 10, 5)
selected = st.sidebar.multiselect("Features for clustering", FEATURES, default=FEATURES)

if len(selected) < 2:
    st.warning("Please select at least 2 features in the sidebar.")
    st.stop()

# ---------- Clustering ----------
X = StandardScaler().fit_transform(df[selected])
model = KMeans(n_clusters=k, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X)
df["Cluster"] = df["Cluster"].astype(str)
score = silhouette_score(X, model.labels_)

# ---------- Header ----------
st.title("🛍️ Customer Segmentation Dashboard")
st.caption("K-Means clustering on the Mall Customers dataset")

c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(df))
c2.metric("Clusters (K)", k)
c3.metric("Silhouette score", f"{score:.3f}")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Cluster plot", "📋 Cluster summary", "📈 Best K", "🗂️ Data"]
)

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Income vs Spending Score")
        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Cluster",
            hover_data=["Age"],
            category_orders={"Cluster": sorted(df["Cluster"].unique())},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("PCA view (2D)")
        pcs = PCA(n_components=2, random_state=42).fit_transform(X)
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        pca_df["Cluster"] = df["Cluster"].values
        fig2 = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            category_orders={"Cluster": sorted(df["Cluster"].unique())},
        )
        st.plotly_chart(fig2, use_container_width=True)

    if len(selected) == 3:
        st.subheader("3D view")
        fig3 = px.scatter_3d(
            df,
            x=selected[0],
            y=selected[1],
            z=selected[2],
            color="Cluster",
            category_orders={"Cluster": sorted(df["Cluster"].unique())},
        )
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Average values per cluster")
    summary = df.groupby("Cluster")[selected].mean().round(1)
    summary["Customers"] = df.groupby("Cluster").size()
    st.dataframe(summary, use_container_width=True)

    st.subheader("Cluster sizes")
    sizes = df["Cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["Cluster", "Customers"]
    st.plotly_chart(
        px.bar(sizes, x="Cluster", y="Customers", color="Cluster"),
        use_container_width=True,
    )

with tab3:
    st.subheader("Elbow method and Silhouette score")
    ks = list(range(2, 11))
    inertias, sils = [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    a, b = st.columns(2)
    a.plotly_chart(
        px.line(x=ks, y=inertias, markers=True,
                labels={"x": "K", "y": "Inertia"}, title="Elbow curve"),
        use_container_width=True,
    )
    b.plotly_chart(
        px.line(x=ks, y=sils, markers=True,
                labels={"x": "K", "y": "Silhouette"}, title="Silhouette score"),
        use_container_width=True,
    )
    st.info(f"Best K by silhouette score: **{ks[sils.index(max(sils))]}**")

with tab4:
    st.subheader("Customers with cluster labels")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download clustered data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="clustered_customers.csv",
        mime="text/csv",
    )

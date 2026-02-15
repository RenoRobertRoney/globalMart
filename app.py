import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# HEADER WITH LOGOS
# ======================================

col_title, col_logo1, col_logo2 = st.columns([6,1,1])

with col_title:
    st.title("🛍️ Customer Segmentation & Marketing Analysis")
    st.markdown("Interactive Machine Learning Dashboard")

with col_logo1:
    st.image("logo.png", use_container_width=True)

with col_logo2:
    st.image("main-logo-alt.webp", use_container_width=True)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to Section:",
    [
        "📂 Upload Data",
        "📊 Data Overview",
        "🔍 PCA Analysis",
        "📉 Clustering Analysis",
        "🌳 Hierarchical Clustering",
        "📢 Marketing Insights",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Machine Learning for Business Intelligence")

# ======================================
# SESSION STATE
# ======================================

if "data" not in st.session_state:
    st.session_state.data = None

# ======================================
# UPLOAD DATA
# ======================================

if page == "📂 Upload Data":

    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("✅ Dataset Loaded Successfully")
        st.dataframe(df.head())

# ======================================
# DATA OVERVIEW
# ======================================

elif page == "📊 Data Overview":

    if st.session_state.data is None:
        st.warning("Please upload dataset first.")
    else:
        df = st.session_state.data

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", df.shape[0])
        col2.metric("Total Features", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature = st.selectbox("Select Feature", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

# ======================================
# PCA ANALYSIS
# ======================================

elif page == "🔍 PCA Analysis":

    if st.session_state.data is None:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA()
        pca.fit(X_scaled)

        explained_variance = np.cumsum(pca.explained_variance_ratio_)

        fig, ax = plt.subplots()
        ax.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Explained Variance")
        st.pyplot(fig)

# ======================================
# CLUSTERING ANALYSIS
# ======================================

elif page == "📉 Clustering Analysis":

    if st.session_state.data is None:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.success("Data Scaled Successfully")

        # -----------------------------
        # ELBOW METHOD
        # -----------------------------

        st.subheader("📉 Elbow Method")

        K_range = range(2, 11)
        inertia = []

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K_range, inertia, marker='o')
        ax_elbow.set_xlabel("Number of Clusters (k)")
        ax_elbow.set_ylabel("Inertia")
        st.pyplot(fig_elbow)

        # -----------------------------
        # FAST SILHOUETTE
        # -----------------------------

        st.subheader("📊 Silhouette Score")

        sil_scores = []

        sample_size = min(2000, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_indices]

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_full = km.fit_predict(X_scaled)
            labels_sample = labels_full[sample_indices]
            sil_scores.append(silhouette_score(X_sample, labels_sample))

        fig_sil, ax_sil = plt.subplots()
        ax_sil.bar(K_range, sil_scores)
        ax_sil.set_xlabel("Number of Clusters (k)")
        ax_sil.set_ylabel("Silhouette Score")
        st.pyplot(fig_sil)

        # -----------------------------
        # FINAL MODEL (k = 2)
        # -----------------------------

        k_selected = 2
        kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        labels = df["Cluster"].values
        silhouette = silhouette_score(X_sample, labels[sample_indices])

        col1, col2 = st.columns(2)
        col1.metric("Selected Clusters", k_selected)
        col2.metric("Silhouette Score", round(silhouette, 3))

        # PCA Visualization
        pca_2 = PCA(n_components=2)
        X_pca_2 = pca_2.fit_transform(X_scaled)

        st.subheader("🎯 Cluster Visualization (2D PCA)")

        fig_cluster, ax_cluster = plt.subplots(figsize=(8,6))
        ax_cluster.scatter(
            X_pca_2[:,0],
            X_pca_2[:,1],
            c=df["Cluster"],
            cmap='viridis',
            alpha=0.7
        )
        ax_cluster.set_xlabel("Principal Component 1")
        ax_cluster.set_ylabel("Principal Component 2")
        st.pyplot(fig_cluster)

        st.session_state.clustered_data = df

# ======================================
# MARKETING INSIGHTS
# ======================================

elif page == "📢 Marketing Insights":

    if "clustered_data" not in st.session_state:
        st.warning("Run Clustering Analysis first.")
    else:
        df = st.session_state.clustered_data

        st.subheader("Final Customer Segment Summary")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        summary = df.groupby("Cluster")[numeric_cols].mean()
        summary["Count"] = df["Cluster"].value_counts()

        st.dataframe(summary)

# ----------------------------------
        # BUSINESS INSIGHTS
        # ----------------------------------

        st.markdown("---")
        st.subheader("📢 Business Insights & Marketing Strategy")

        # Overall averages for comparison
        overall_income = df["Annual_Income_K"].mean() if "Annual_Income_K" in df.columns else 0
        overall_spending = df["Spending_Score"].mean() if "Spending_Score" in df.columns else 0

        for cluster in summary.index:

            st.markdown(f"### 🔹 Cluster {cluster}")

            cluster_income = summary.loc[cluster].get("Annual_Income_K", 0)
            cluster_spending = summary.loc[cluster].get("Spending_Score", 0)
            cluster_size = summary.loc[cluster]["Count"]

            st.write(f"• Customers in this segment: **{int(cluster_size)}**")
            st.write(f"• Avg Income: **{round(cluster_income,2)}**")
            st.write(f"• Avg Spending Score: **{round(cluster_spending,2)}**")

            if cluster_income > overall_income and cluster_spending > overall_spending:
                st.success("💎 High Income & High Spending → Premium customers. Focus on luxury campaigns, loyalty rewards, and exclusive memberships.")

            elif cluster_income > overall_income and cluster_spending <= overall_spending:
                st.info("🏦 High Income but Low Spending → Upsell through personalized offers, premium bundles, and targeted engagement.")

            elif cluster_income <= overall_income and cluster_spending > overall_spending:
                st.warning("🛍️ Moderate Income but High Spending → Offer discounts, cashback deals, and seasonal promotions.")

            else:
                st.error("💰 Low Income & Low Spending → Cost-sensitive segment. Use budget-friendly campaigns and avoid heavy ad spending.")

            st.markdown("---")

# ======================================
# HIERARCHICAL CLUSTERING
# ======================================

elif page == "🌳 Hierarchical Clustering":

    if st.session_state.data is None:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        sample_size = st.slider("Sample Size for Dendrogram", 100, 2000, 500)

        if st.button("Generate Dendrogram"):
            sample_data = X_scaled[:sample_size]
            linked = linkage(sample_data, method='ward')

            fig5 = plt.figure(figsize=(10,5))
            dendrogram(linked)
            plt.title("Dendrogram")
            st.pyplot(fig5)

# ======================================
# ABOUT SECTION
# ======================================

elif page == "ℹ️ About":

    st.header("About This Project")

    st.subheader("👨‍💻 Team Members")
    st.write("- Nandhana Anilkumar")
    st.write("LinkedIn: https://www.linkedin.com/in/nandhana-anilkumar-b55661333/")
    st.write("- Reno Robert Roney")
    st.write("LinkedIn: https://www.linkedin.com/in/reno-robert-roney-690989323/")
    st.write("- Edwin K Johnson")
    st.write("LinkedIn: https://www.linkedin.com/in/edwin-k-johnson-087500308/")
    st.write("- Joyal Raju")
    st.write("LinkedIn: https://www.linkedin.com/in/joyal-raju-78a3a136a/")
    st.write("- Chackochan Siju ")
    st.write("LinkedIn: https://www.linkedin.com/in/chackochan-siju-2b35b8333/")


    st.subheader("🛠️ Tech Stack")
    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - PCA
    - KMeans
    - Hierarchical Clustering
    - Streamlit
    """)

    st.subheader("⚙️ How This App Works")
    st.write("""
    1. Upload customer dataset.
    2. Data is standardized.
    3. PCA reduces dimensionality.
    4. KMeans clusters customers.
    5. Elbow & Silhouette evaluate cluster quality.
    6. Marketing insights are generated from cluster averages.
    """)

    st.success("Built for Machine Learning for Business Intelligence")

# app.py
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import umap.umap_ as umap_model

from sklearn.manifold import TSNE
import pydeck as pdk
import altair as alt
from mlflow.tracking import MlflowClient
from streamlit_folium import st_folium
import folium
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union
import joblib

st.set_page_config(layout="wide", page_title="PatrolQ — Crime Explorer")

@st.cache_data(show_spinner=False)
def load_data(nrows=None):
    CSV_URL = "https://your-storage-link/sample_df.csv"   # <-- your uploaded file URL
    df = pd.read_csv(CSV_URL, parse_dates=["Date", "Updated On"], low_memory=False)
    
    if nrows:
        df = df.sample(nrows, random_state=42).reset_index(drop=True)
    
    return df


@st.cache_data
def sample_coords(df, n=20000):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42).reset_index(drop=True)

@st.cache_data
def prepare_features(df, feature_columns):
    X = df[feature_columns].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


@st.cache_data
def compute_umap(X, n_components=2):
    reducer = umap_model.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)


@st.cache_data
def compute_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca

@st.cache_data
def compute_tsne(X, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate='auto', random_state=42)
    return tsne.fit_transform(X)

def cluster_convex_hulls(df, label_col="cluster"):
    # compute convex hull per cluster for boundaries (ignores noise -1)
    hulls = []
    for lab in sorted(df[label_col].unique()):
        if lab == -1:
            continue
        pts = df[df[label_col] == lab][["Longitude", "Latitude"]].dropna().values
        if len(pts) < 3:
            continue
        hull = MultiPoint([Point(xy) for xy in pts]).convex_hull
        hulls.append((lab, hull))
    return hulls

# --- Sidebar ---
st.sidebar.title("PatrolQ — Controls")
data_path = st.sidebar.text_input("CSV path", "sample_df.csv")
rows = st.sidebar.number_input("Load sample rows (0 = full)", min_value=0, value=200000, step=10000)
page = st.sidebar.radio("Page", ["Overview", "Map", "Temporal", "Dimensionality Reduction", "Monitoring"])
feature_columns_default = ["PC1","PC2","PC3","lat_norm","lon_norm"]

# --- Load data ---
with st.spinner("Loading dataset..."):
    df = load_data(data_path, None if rows==0 else rows)

st.sidebar.markdown(f"**Rows loaded:** {len(df):,}")

# --- Overview page ---
if page == "Overview":
    st.title("PatrolQ — Dataset Overview")
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("Top Primary Types")
        st.table(df["Primary Type"].value_counts().head(10).rename_axis("Primary Type").reset_index(name="count"))
        st.header("Spatial sample (20k)")
        sample20 = sample_coords(df, n=20000)
        st.map(sample20.rename(columns={"Latitude":"lat","Longitude":"lon"})[["lat","lon"]])
    with col2:
        st.header("Quick Stats")
        st.metric("Total incidents", f"{len(df):,}")
        st.metric("Unique crime types", df["Primary Type"].nunique())
        st.metric("Unique Beats", df["Beat"].nunique())
        st.markdown("---")
        st.markdown("### Filters")
        chosen_crimes = st.multiselect("Filter Primary Type (top 10)", df["Primary Type"].value_counts().head(10).index.tolist())
        if chosen_crimes:
            st.write(df[df["Primary Type"].isin(chosen_crimes)].shape)

# --- Map page ---
elif page == "Map":
    st.title("Geographic Heatmap & Cluster Boundaries")
    st.sidebar.header("Map options")
    show_clusters = st.sidebar.checkbox("Show cluster convex hulls (needs 'cluster' column)", value=True)
    heatmap_sample = st.sidebar.number_input("Heatmap sample size", min_value=2000, max_value=50000, value=20000, step=1000)
    map_df = sample_coords(df, heatmap_sample)

    lat_center = map_df["Latitude"].median()
    lon_center = map_df["Longitude"].median()

    st.subheader("Folium heatmap")
    fol = folium.Map(location=[lat_center, lon_center], zoom_start=11)
    #importing the map
    from folium.plugins import HeatMap
    
    heat_data = map_df[["Latitude","Longitude"]].dropna().values.tolist()
    HeatMap(heat_data, radius=10, blur=12, max_zoom=13).add_to(fol)

    # draw convex hulls if available
    if show_clusters and "kmeans_cluster" in df.columns:
        hulls = cluster_convex_hulls(df, label_col="kmeans_cluster")
        for lab, hull in hulls:
            coords = [[y, x] for x, y in hull.exterior.coords]  # folium uses lat,lon order
            folium.Polygon(locations=coords, color="red", fill=False, weight=2, popup=f"kmeans_cluster {lab}").add_to(fol)
    st_data = st_folium(fol, width=1000)

    st.markdown("### PyDeck sample (colored by cluster if present)")
    pd_sample = sample_coords(df if "kmeans_cluster" in df.columns else map_df, 20000)
    color_col = "kmeans_cluster" if "kmeans_cluster" in pd_sample.columns else None
    if color_col:
        pd_sample["color_code"] = (pd_sample[color_col].astype(int) - pd_sample[color_col].min()).astype(int) % 10
    else:
        pd_sample["color_code"] = 1

    layer = pdk.Layer(
        "HexagonLayer",
        data=pd_sample,
        get_position=["Longitude", "Latitude"],
        radius=200,
        elevation_scale=50,
        elevation_range=[0, 3000],
        pickable=True,
        extruded=True
    )
    view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=11, pitch=40)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

# --- Temporal page ---
elif page == "Temporal":
    st.title("Temporal Pattern Analysis")
    st.sidebar.header("Temporal options")
    freq = st.sidebar.selectbox("Aggregation frequency", ["H", "D", "W", "M"], index=1)
    time_col = "Date"
    df_time = df.set_index(time_col)
    ts = df_time.resample(freq).size().rename("count").reset_index()
    
    chart = alt.Chart(ts).mark_line().encode(x=time_col, y="count").properties(width=900, height=350)
    st.altair_chart(chart, use_container_width=True)

    st.header("By Day of Week and Hour")
    dow = df.groupby(["DayOfWeek"]).size().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).rename("count").reset_index()
    st.bar_chart(dow.set_index("DayOfWeek"))

    # heatmap hour vs dow
    pivot = df.pivot_table(index="Hour", columns="DayOfWeek", values="ID", aggfunc="count").fillna(0)
    st.write("Hour vs DayOfWeek heatmap")
    st.dataframe(pivot.style.background_gradient(cmap="magma"))

# --- DimRed page ---
elif page == "Dimensionality Reduction":
    st.title("Dimensionality Reduction Viewer")
    st.sidebar.header("DimRed options")
    #features
    features = st.sidebar.multiselect("Feature columns", feature_columns_default, feature_columns_default)
    sample_size = st.sidebar.slider("Sample size for Dimensionality Reduction", min_value=2000, max_value=50000, value=20000, step=1000)
    method_dr = st.sidebar.selectbox("Method", ["UMAP","PCA","t-SNE"], index=0)

    dr_df = sample_coords(df, sample_size).reset_index(drop=True)
    Xs, scaler = prepare_features(dr_df, features)

    if method_dr == "UMAP":
        emb = compute_umap(Xs, n_components=2)
    elif method_dr == "PCA":
        emb, pca_model = compute_pca(Xs, n_components=2)
    else:
        emb = compute_tsne(Xs, perplexity=30, n_iter=1000)

    dr_df["dim1"] = emb[:,0]
    dr_df["dim2"] = emb[:,1]

    color_by = st.selectbox("Color by", ["Primary Type","kmeans_cluster"], index=0 if "Primary Type" in dr_df.columns else 1)
    if color_by in dr_df.columns:
        chart = alt.Chart(dr_df).mark_circle(size=8).encode(
            x="dim1", y="dim2", color=color_by, tooltip=["Primary Type","kmeans_cluster","PC1"]
        ).interactive().properties(width=900, height=600)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No such column to color by in sampled data.")

# --- Monitoring page ---
elif page == "Monitoring":
    st.title("MLflow Monitoring & Model Registry")
    st.sidebar.header("MLflow Config")
    tracking_uri = st.sidebar.text_input("MLflow tracking URI", "http://localhost:5004")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    exp_name = st.sidebar.text_input("Experiment name", "crime_clustering")
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        st.warning(f"Experiment {exp_name} not found in MLflow tracking server at {tracking_uri}")
    else:
        st.write("Experiment:", exp.name, "id:", exp.experiment_id)
        runs = client.search_runs(exp.experiment_id, order_by=["metrics.silhouette DESC"], max_results=20)
        run_table = []
        for r in runs:
            run_table.append({
                "run_id": r.info.run_id,
                "start_time": pd.to_datetime(r.info.start_time, unit='ms'),
                "silhouette": r.data.metrics.get("silhouette"),
                "dbi": r.data.metrics.get("davies_bouldin"),
                "model_name": ( r.data.tags.get("mlflow.runName") or "Unknown"),
                
                "method": r.data.params.get("method"),
                "k": r.data.params.get("k")
            })
        st.dataframe(pd.DataFrame(run_table))

        if st.button("Open MLflow UI (localhost port 5000)"):
            st.write("Open your MLflow UI at http://localhost:5004")

        # model registry list
        try:
            regs = client.search_registered_models()
            reg_names = [m.name for m in regs]
            st.write("Registered models:", reg_names)
        except Exception as e:
            st.write("No registry available / not reachable", e)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("PatrolQ — Streamlit app\nBuilt for interactive crime clustering & analysis")


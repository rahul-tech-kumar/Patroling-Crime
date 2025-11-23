# 🛡️ PatrolIQ — Smart Safety Analytics Platform

Loads patrol datasets, performs PCA/UMAP/t-SNE, applies KMeans, DBSCAN, and Hierarchical clustering, and visualizes crime patterns interactively using Streamlit.

# 🚀 Overview

- PatrolIQ is an interactive crime-analytics dashboard built with Streamlit, designed to analyze large-scale patrol datasets and uncover crime patterns using modern clustering and dimensionality-reduction techniques.

- The app integrates:

- 📊 PCA, UMAP, t-SNE for dimensionality reduction

- 🤖 KMeans, DBSCAN, Agglomerative (Hierarchical) clustering

- 🌍 Geo-visualization using Folium & PyDeck

- 📝 MLflow for tracking experiments and models

- ⚡ Streamlit caching for performance

- 📡 Ability to load dataset from cloud storage (required for large files)

- ✨ Features
# 🔍 Data Exploration

- Dataset overview and summary statistics

- Top crime categories

- Sampling utilities for large datasets

# 🗺️ Geospatial Analysis

- Heatmaps with Folium

- Cluster convex hulls

- 3D hexagonal density visualization with PyDeck

# 🧠 Clustering Algorithms

- KMeans

- DBSCAN

- Hierarchical (Agglomerative)

- Metrics: silhouette score, Davies–Bouldin index

# 📉 Dimensionality Reduction

- PCA (2D, 3D)

- UMAP (2D)

- t-SNE

# 📁 MLflow Integration

- Track clustering runs

- Compare metrics

- View best models

- Register models in Model Registry

# 🏗️ Project Structure
PatrolIQ/
│── app.py                 
│── sample_df.csv           
│── requirements.txt
│── README.md
|-- patroling.py

# 🧪 Running Locally
- Sreamlit App (http://localhost:8501/)
- MLFlow Run (http://127.0.0.1:5004/#/)

# Added Screenshot
## ML Flow
<img width="1920" height="1080" alt="Screenshot (161)" src="https://github.com/user-attachments/assets/52a71ff4-b225-4da6-b0ab-6a68dd5c6cd8" />
## Streamlit
<img width="1920" height="1080" alt="Screenshot (160)" src="https://github.com/user-attachments/assets/79bff9d6-6281-4530-ac8c-c9bbfbf3348e" />
<img width="1920" height="1080" alt="Screenshot (162)" src="https://github.com/user-attachments/assets/0bd1c403-f52d-4c8a-bd87-9a7afcb984c0" />
<img width="1920" height="1080" alt="Screenshot (163)" src="https://github.com/user-attachments/assets/f0f13a9a-b7ff-4df7-8e50-32881a4feeac" />
<img width="1920" height="1080" alt="Screenshot (164)" src="https://github.com/user-attachments/assets/cad60a1a-d808-44e1-b988-6b6e12737357" />




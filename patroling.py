# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import mlflow
# import mlflow.sklearn
# import warnings
# warnings.filterwarnings("ignore")

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score, davies_bouldin_score
# from sklearn.decomposition import PCA
# import seaborn as sns
# import os

# # -------------------- LOAD DATASET -----------------------------
# print("Loading Dataset")
# df = pd.read_csv("sample_df.csv")

# # -------------------- STANDARDIZATION --------------------------
# print("Standardizing the DATA")
# feature_columns = ["PC1", "PC2", "PC3", "lat_norm", "lon_norm"]
# X = StandardScaler().fit_transform(df[feature_columns].values)

# # ---------------- MLflow Setup --------------------------------
# mlflow.set_tracking_uri("http://localhost:5004")
# mlflow.set_experiment("crime_clustering")

# # ---------------- Models ---------------------------------------
# clustering_models = [
#     ("KMeans",
#      {"n_clusters": 5, "init": "k-means++", "n_init": "auto", "random_state": 42},
#      KMeans()),

#     # ("DBSCAN",
#     #  {"eps": 0.25, "min_samples": 10, "metric": "euclidean", "n_jobs": -1},
#     #  DBSCAN()),

#     ("Agglomerative Clustering",
#      {"n_clusters": 5, "linkage": "ward"},
#      AgglomerativeClustering())
# ]

# clustering_results = []

# print("Running Models...\n")

# # =====================================================================
# #                        MAIN MODEL LOOP
# # =====================================================================
# for model_name, params, model in clustering_models:
#     with mlflow.start_run(run_name=model_name):

#         print(f"🚀 Running Clustering: {model_name}...\n")

#         # Create artifact folder
#         model_dir = f"artifacts/{model_name.replace(' ', '_')}"
#         os.makedirs(model_dir, exist_ok=True)

#         # Set model parameters
#         model.set_params(**params)

#         # ==============================================================
#         #                       DBSCAN (SPECIAL CASE)
#         # ==============================================================
#         # if model_name == "DBSCAN":
#         #     print("========== Running DBSCAN ==========\n")

#         #     try:
#         #         labels = model.fit_predict(X)
#         #         df["cluster"] = labels
#         #         mlflow.log_param("dbscan_memory_status", "OK")

#         #     except MemoryError:
#         #         print("❌ MEMORY ERROR in DBSCAN — Forcing all labels to -1")
#         #         labels = np.array([-1] * len(X))
#         #         df["cluster"] = labels
#         #         mlflow.log_param("dbscan_memory_status", "MemoryError")

#         # ==============================================================
#         #                       OTHER MODELS
#         # ==============================================================
#         # else:
#         print(f"========== Running {model_name} ==========\n")

#         labels = model.fit_predict(X)
#         df["cluster"] = labels

#         # ==============================================================
#         #                           METRICS
#         # ==============================================================
#         sample_size = 500 if model_name == "DBSCAN" else 10000

#         if len(X) > sample_size:
#             idx = np.random.choice(len(X), sample_size, replace=False)
#             X_sample = X[idx]
#             labels_sample = labels[idx]
#         else:
#             X_sample = X
#             labels_sample = labels

#         # handle invalid labels case
#         if len(set(labels_sample)) > 1 and not set(labels_sample) == {-1}:
#             sil = silhouette_score(X_sample, labels_sample)
#             dbi = davies_bouldin_score(X_sample, labels_sample)
#         else:
#             sil = -1
#             dbi = 999

#         print(f"✔ Silhouette Score: {sil}")
#         print(f"✔ DBI Score: {dbi}")

#         mlflow.log_metric("silhouette", sil)
#         mlflow.log_metric("davies_bouldin", dbi)
#         mlflow.log_param("metric_sample_size", sample_size)

#         # ==============================================================
#         #                   SAVE CLUSTERED DATA
#         # ==============================================================
#         csv_path = os.path.join(model_dir, "clustered_output.csv")
#         df.to_csv(csv_path, index=False)
#         mlflow.log_artifact(csv_path)

#         # ==============================================================
#         #                   PCA 2D PLOT
#         # ==============================================================
#         pca = PCA(n_components=2)
#         reduced = pca.fit_transform(X)

#         df_plot = pd.DataFrame({
#             "PC1": reduced[:, 0],
#             "PC2": reduced[:, 1],
#             "cluster": labels
#         })

#         plt.figure(figsize=(7, 6))
#         sns.scatterplot(
#             data=df_plot,
#             x="PC1", y="PC2",
#             hue="cluster",
#             palette="tab10",
#             s=40
#         )
#         plt.title(f"{model_name} - PCA Cluster Plot")

#         plot_path = os.path.join(model_dir, "cluster_plot.png")
#         plt.savefig(plot_path)
#         mlflow.log_artifact(plot_path)
#         plt.close()

#         # ==============================================================
#         #                   LOG MODEL IN MLFLOW
#         # ==============================================================
#         try:
#             mlflow.sklearn.log_model(
#                 sk_model=model,
#                 artifact_path="clustering_model",
#                 registered_model_name=model_name.replace(" ", "_") + "_Clustering"
#             )
#         except Exception as e:
#             mlflow.log_param("model_log_status", f"Failed: {str(e)}")

#         # Save run result
#         clustering_results.append({
#             "Model": model_name,
#             "Silhouette": sil,
#             "DBI": dbi,
#             "run_id": mlflow.active_run().info.run_id
#         })

# # =====================================================================
# #                       END
# # =====================================================================
# print("\n\n🏁 All Models Completed Successfully!")

# print("\n🎉 Clustering Completed Successfully!")

# # ---------------------------------------------------------------------


# # ---------------------------------------------------------------------
# #  BEST MODEL SELECTION (Based on Silhouette Score)
# # ---------------------------------------------------------------------
# print("\n🔍 Selecting Best Clustering Model Based on Silhouette Score...")

# results_df = pd.DataFrame(clustering_results)
# best_row = results_df.loc[results_df["silhouette"].idxmax()]

# best_model_name = best_row["Model"]
# best_run_id = best_row["run_id"]
# best_silhouette = best_row["silhouette"]

# print("\n🏆 Best Model Selected:")
# print(f"Model: {best_model_name}")
# print(f"Run ID: {best_run_id}")
# print(f"Silhouette Score: {best_silhouette:.4f}")

# # ---------------------------------------------------------------------
# #  REGISTER BEST MODEL INTO MLflow MODEL REGISTRY
# # ---------------------------------------------------------------------
# from mlflow.tracking import MlflowClient
# client = MlflowClient()

# model_uri = f"runs:/{best_run_id}/model"
# registered_name = best_model_name.replace(" ", "_") + "_Clustering_Production"

# registered_model = mlflow.register_model(
#     model_uri=model_uri,
#     name=registered_name
# )

# print("\n🚀 Best clustering model registered to MLflow Model Registry!")
# print(f"Registered Name: {registered_name}")
# print("You can now view, version, or deploy it via MLflow UI.")


# ================= CLEANED CLUSTERING SCRIPT =====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import seaborn as sns
import os

# -------------------- LOAD DATASET -----------------------------
print("-----------")
df = pd.read_csv("sample_df.csv")

# -------------------- STANDARDIZATION --------------------------
feature_columns = ["PC1", "PC2", "PC3", "lat_norm", "lon_norm"]
X = StandardScaler().fit_transform(df[feature_columns].values)

# ---------------- MLflow Setup --------------------------------
mlflow.set_tracking_uri("http://localhost:5004")
mlflow.set_experiment("crime_clustering")

# ---------------- Models ---------------------------------------
clustering_models = [
    ("KMeans",
     {"n_clusters": 5, "init": "k-means++", "n_init": "auto", "random_state": 42},
     KMeans()),

    ("Agglomerative Clustering",
     {"n_clusters": 5, "linkage": "ward"},
     AgglomerativeClustering())
]

clustering_results = []

# ===================== MAIN LOOP ===============================
for model_name, params, model in clustering_models:
    with mlflow.start_run(run_name=model_name):
        print(f"🚀 Running Clustering: {model_name}...")

        # create artifact folder
        model_dir = f"artifacts/{model_name.replace(' ', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        # set parameters
        model.set_params(**params)

        # PCA for Agglomerative to reduce memory usage
        if model_name == "Agglomerative Clustering":
            pca_reduced = PCA(n_components=5)
            X_model = pca_reduced.fit_transform(X)
        else:
            X_model = X

        # fit and predict
        labels = model.fit_predict(X_model)
        df["cluster"] = labels

        # compute metrics (sampled)
        sample_size = 5000
        if len(X_model) > sample_size:
            idx = np.random.choice(len(X_model), sample_size, replace=False)
            X_sample = X_model[idx]
            labels_sample = labels[idx]
        else:
            X_sample = X_model
            labels_sample = labels

        if len(set(labels_sample)) > 1 and not set(labels_sample) == {-1}:
            sil = silhouette_score(X_sample, labels_sample)
            dbi = davies_bouldin_score(X_sample, labels_sample)
        else:
            sil = -1
            dbi = 999

        print(f"✔ Silhouette Score: {sil}")
        print(f"✔ DBI Score: {dbi}")

        mlflow.log_metric("silhouette", sil)
        mlflow.log_metric("davies_bouldin", dbi)

        # save clustered data
        csv_path = os.path.join(model_dir, "clustered_output.csv")
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # 2D PCA plot
        pca_2d = PCA(n_components=2)
        reduced_2d = pca_2d.fit_transform(X)
        df_plot = pd.DataFrame({
            "PC1": reduced_2d[:, 0],
            "PC2": reduced_2d[:, 1],
            "cluster": labels
        })

        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df_plot, x="PC1", y="PC2",
            hue="cluster", palette="tab10", s=40
        )
        plt.title(f"{model_name} - PCA Cluster Plot")
        plot_path = os.path.join(model_dir, "cluster_plot.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # log model
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="clustering_model",
                registered_model_name=model_name.replace(" ", "_") + "_Clustering"
            )
        except Exception as e:
            mlflow.log_param("model_log_status", f"Failed: {str(e)}")

        # save run info
        clustering_results.append({
            "Model": model_name,
            "Silhouette": sil,
            "DBI": dbi,
            "run_id": mlflow.active_run().info.run_id
        })

# ===================== BEST MODEL SELECTION =====================
results_df = pd.DataFrame(clustering_results)
best_row = results_df.loc[results_df["Silhouette"].idxmax()]

best_model_name = best_row["Model"]
best_run_id = best_row["run_id"]
best_silhouette = best_row["Silhouette"]

print(f"\n🏆 Best Model: {best_model_name}, Run ID: {best_run_id}, Silhouette: {best_silhouette:.4f}")

# register best model
model_uri = f"runs:/{best_run_id}/model"
registered_name = best_model_name.replace(" ", "_") + "_Clustering_Production"
registered_model = mlflow.register_model(model_uri=model_uri, name=registered_name)

print(f"\n🚀 Registered Best Model: {registered_name}")

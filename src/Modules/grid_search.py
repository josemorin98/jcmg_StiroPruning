import umap
import hdbscan
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd

class GridSearchManager(BaseEstimator, ClusterMixin):
    def __init__(self, n_neighbors=15, n_components=5, min_cluster_size=5):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        self.umap_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric='cosine',
            n_jobs=-1
        ).fit(X)
        X_umap = self.umap_.transform(X)
        self.hdbscan_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(X_umap)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.hdbscan_.labels_

    def predict(self, X):
        X_umap = self.umap_.transform(X)
        return self.hdbscan_.approximate_predict(X_umap)[0]

    def score(self, X, y=None):
        labels = self.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        return -(n_noise + abs(n_clusters - 10))  # Ajusta según tu criterio

    def run_grid_search(self, embeddings, param_grid, modelos_dir="../test/Modelos", name="modelo"):
        """
        Ejecuta GridSearchCV sobre los parámetros dados y guarda el mejor modelo y resultados.
        """
        grid = GridSearchCV(
            self,
            param_grid,
            scoring=None,  # Usa el método score del estimador
            refit=True,
            cv=[(slice(None), slice(None))],  # Sin validación cruzada real
            verbose=2,
            n_jobs=8
        )
        grid.fit(embeddings)
        print("Mejores parámetros:", grid.best_params_)
        print("Mejor score:", grid.best_score_)

        # Guardar el mejor modelo
        import os
        os.makedirs(modelos_dir, exist_ok=True)
        with open(f"{modelos_dir}/umap_hdbscan_gridsearch.pkl", "wb") as f:
            pickle.dump(grid.best_estimator_, f)

        # Guardar resultados del grid search con el nombre del modelo
        results_df = pd.DataFrame(grid.cv_results_)
        csv_path = f"{modelos_dir}/gridsearch_{name}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Resultados y modelo guardados en {modelos_dir} (archivo: {csv_path})")

        return grid.best_params_, grid.best_score_, results_df

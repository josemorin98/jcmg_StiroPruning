"""
Estimadores personalizados para UMAP y HDBSCAN compatibles con GridSearchCV
"""

import numpy as np
import umap
import hdbscan
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
from sklearn.metrics import silhouette_score


class UMAPEstimator(BaseEstimator, TransformerMixin):
    """
    Estimador UMAP compatible con GridSearchCV de scikit-learn.
    """
    
    def __init__(self, n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Ajusta el modelo UMAP a los datos."""
        self.umap_model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=3
        ).fit(X)
        return self
    
    def transform(self, X):
        """Transforma los datos usando el modelo UMAP ajustado."""
        return self.umap_model_.transform(X)
    
    def fit_transform(self, X, y=None):
        """Ajusta el modelo y transforma los datos en un solo paso."""
        return self.fit(X).transform(X)
    
    def score(self, X, y=None):
        """Puntuación basada en la preservación de la estructura local."""
        X_transformed = self.transform(X)
        # Usar negativo de la varianza como score (menor varianza = mejor)
        return -np.var(X_transformed)


class HDBSCANEstimator(BaseEstimator, ClusterMixin):
    """
    Estimador HDBSCAN compatible con GridSearchCV de scikit-learn.
    """
    
    def __init__(self, min_cluster_size=10, min_samples=10, cluster_selection_epsilon=0.0, 
                 metric='euclidean', cluster_selection_method='eom'):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        
    def fit(self, X, y=None):
        """Ajusta el modelo HDBSCAN a los datos."""
        self.hdbscan_model_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method
        ).fit(X)
        
        self.labels_ = self.hdbscan_model_.labels_
        self.probabilities_ = self.hdbscan_model_.probabilities_
        return self
    
    def fit_predict(self, X, y=None):
        """Ajusta el modelo y retorna las etiquetas de clustering."""
        return self.fit(X).labels_
    
    def score(self, X, y=None):
        """Puntuación basada en la calidad del clustering."""
        if not hasattr(self, 'labels_'):
            self.fit(X)
        
        # Si hay menos de 2 clusters únicos, retornar puntuación muy baja
        unique_labels = np.unique(self.labels_)
        if len(unique_labels) < 2:
            return -1.0
        
        # Calcular silhouette score solo para puntos no-ruido
        mask = self.labels_ != -1
        if np.sum(mask) < 2:
            return -1.0
        
        try:
            silhouette = silhouette_score(X[mask], self.labels_[mask])
            # Penalizar por alta proporción de ruido
            noise_penalty = np.sum(self.labels_ == -1) / len(self.labels_)
            return silhouette - noise_penalty
        except:
            return -1.0


class UMAPHDBSCANEstimator(BaseEstimator, ClusterMixin):
    """
    Estimador que combina UMAP + HDBSCAN para GridSearchCV conjunto.
    """
    
    def __init__(self, n_neighbors=15, n_components=2, min_cluster_size=10, min_samples=10, random_state=42):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """Ajusta UMAP + HDBSCAN a los datos."""
        # Aplicar UMAP
        self.umap_model_ = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state,
            n_jobs=1
        ).fit(X)
        
        umap_embeddings = self.umap_model_.transform(X)
        
        # Aplicar HDBSCAN
        self.hdbscan_model_ = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(umap_embeddings)
        
        self.labels_ = self.hdbscan_model_.labels_
        self.probabilities_ = self.hdbscan_model_.probabilities_
        
        return self
    
    def fit_predict(self, X, y=None):
        """Ajusta el modelo y devuelve las etiquetas de clustering."""
        return self.fit(X).labels_
    
    def score(self, X, y=None):
        """Función de puntuación para GridSearchCV."""
        if not hasattr(self, 'labels_'):
            self.fit(X)
        
        # Calcular el costo (proporción de puntos con baja probabilidad)
        prob_threshold = 0.05
        total_num = len(self.labels_)
        cost = np.count_nonzero(self.probabilities_ < prob_threshold) / total_num
        
        # Retornar el negativo del costo (GridSearchCV maximiza el score)
        return -cost

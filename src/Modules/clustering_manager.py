import os
import warnings
# Configuración para ocultar warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ocultar warnings de TensorFlow
warnings.filterwarnings('ignore')  # Ocultar otros warnings
import umap
import hdbscan
import numpy as np
import pandas as pd
from tqdm import trange
from functools import partial
from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import make_scorer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from .estimators import UMAPEstimator, HDBSCANEstimator, UMAPHDBSCANEstimator
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns   


class ClusteringManager:
    """
    Clase para gestionar la reducción de dimensionalidad y clustering sobre embeddings.
    Permite realizar búsquedas aleatorias de hiperparámetros y evaluar los resultados.
    """

    def __init__(self, random_state=42, model="use"):
        """
        Inicializa el gestor de clustering.

        Parámetros:
            random_state (int): Semilla para reproducibilidad.
        """
        self.model = model
        if model not in ["use", "st1", "st2", "st3"]:
            raise ValueError(f"Modelo no soportado: {model}")
        self.random_state = random_state

    def _calculate_dbcv(self, embeddings, labels):
        """
        Calcula la métrica DBCV (Density-Based Cluster Validation) de manera simplificada.
        Implementación propia basada en la separación inter-cluster e intra-cluster.
        
        Parámetros:
            embeddings (np.ndarray): Embeddings utilizados para clustering.
            labels (np.ndarray): Etiquetas de clustering.
            
        Retorna:
            float: Valor DBCV aproximado o None si no se puede calcular.
        """
        try:
            # Filtrar puntos de ruido (-1) para DBCV
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                print("Warning: No hay suficientes puntos no-ruido para calcular DBCV")
                return None
            
            filtered_embeddings = embeddings[non_noise_mask]
            filtered_labels = labels[non_noise_mask]
            
            # Verificar que hay al menos 2 clusters
            unique_labels = np.unique(filtered_labels)
            if len(unique_labels) < 2:
                print("Warning: Se necesitan al menos 2 clusters para calcular DBCV")
                return None
            
            # Implementación simplificada de DBCV
            # Basada en densidad local y separación entre clusters
            
            # Calcular distancias dentro de cada cluster (cohesión)
            intra_cluster_distances = []
            for label in unique_labels:
                cluster_mask = filtered_labels == label
                cluster_points = filtered_embeddings[cluster_mask]
                
                if len(cluster_points) > 1:
                    # Distancia promedio dentro del cluster
                    distances = euclidean_distances(cluster_points)
                    # Tomar el triángulo superior excluyendo la diagonal
                    upper_triangle = np.triu(distances, k=1)
                    non_zero_distances = upper_triangle[upper_triangle > 0]
                    if len(non_zero_distances) > 0:
                        avg_intra_dist = np.mean(non_zero_distances)
                        intra_cluster_distances.append(avg_intra_dist)
            
            # Calcular distancias entre clusters (separación)
            inter_cluster_distances = []
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    cluster1_points = filtered_embeddings[filtered_labels == label1]
                    cluster2_points = filtered_embeddings[filtered_labels == label2]
                    
                    # Distancia promedio entre clusters
                    distances = euclidean_distances(cluster1_points, cluster2_points)
                    avg_inter_dist = np.mean(distances)
                    inter_cluster_distances.append(avg_inter_dist)
            
            if len(intra_cluster_distances) == 0 or len(inter_cluster_distances) == 0:
                return None
            
            # Calcular DBCV simplificado
            avg_intra = np.mean(intra_cluster_distances)
            avg_inter = np.mean(inter_cluster_distances)
            
            # DBCV = (separación - cohesión) / max(separación, cohesión)
            # Valores más altos indican mejor clustering
            if max(avg_inter, avg_intra) > 0:
                dbcv_score = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
            else:
                dbcv_score = 0.0
            
            return dbcv_score
            
        except Exception as e:
            print(f"Error calculando DBCV: {str(e)}")
            return None

    def _save_dbcv_results(self, method_name, best_params, dbcv_score, n_clusters, noise_ratio, modelos_dir):
        """
        Guarda los resultados de DBCV en un CSV específico.
        
        Parámetros:
            method_name (str): Nombre del método (random, bayesian, grid_search)
            best_params (dict): Mejores parámetros encontrados
            dbcv_score (float): Score DBCV calculado
            n_clusters (int): Número de clusters
            noise_ratio (float): Proporción de ruido
            modelos_dir (str): Directorio donde guardar
        """
        try:
            os.makedirs(modelos_dir, exist_ok=True)
            
            dbcv_results = {
                'method': method_name,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'n_neighbors': best_params.get('n_neighbors', None),
                'n_components': best_params.get('n_components', None),
                'min_cluster_size': best_params.get('min_cluster_size', None),
                'min_samples': best_params.get('min_samples', None),
                'min_dist': best_params.get('min_dist', None),
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'dbcv_score': dbcv_score,
                'model': self.model
            }
            
            csv_path = os.path.join(modelos_dir, f"dbcv_results_{method_name}.csv")
            df = pd.DataFrame([dbcv_results])
            
            # Si el archivo existe, agregarlo; si no, crearlo
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                df_combined = pd.concat([df_existing, df], ignore_index=True)
                df_combined.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, index=False)
            
            print(f"Resultados DBCV guardados en {csv_path}")
            
        except Exception as e:
            print(f"Error guardando resultados DBCV: {str(e)}")

    def _save_consolidated_dbcv_results(self, modelos_dir):
        """
        Crea un archivo consolidado con todos los resultados DBCV de los diferentes métodos.
        """
        try:
            consolidated_data = []
            
            # Buscar todos los archivos de resultados DBCV
            methods = ['random_search', 'bayesian_search', 'separate_grid_search']
            
            for method in methods:
                csv_path = os.path.join(modelos_dir, f"dbcv_results_{method}.csv")
                if os.path.exists(csv_path):
                    df_method = pd.read_csv(csv_path)
                    consolidated_data.append(df_method)
            
            if consolidated_data:
                # Combinar todos los DataFrames
                consolidated_df = pd.concat(consolidated_data, ignore_index=True)
                
                # Ordenar por DBCV score (descendente - mejores primero)
                consolidated_df = consolidated_df.sort_values('dbcv_score', ascending=False, na_last=True)
                
                # Guardar archivo consolidado
                consolidated_path = os.path.join(modelos_dir, "dbcv_results_consolidated.csv")
                consolidated_df.to_csv(consolidated_path, index=False)
                
                print(f"Resultados DBCV consolidados guardados en {consolidated_path}")
                
                # Mostrar resumen
                print("\n=== RESUMEN DE RESULTADOS DBCV ===")
                for method in consolidated_df['method'].unique():
                    method_data = consolidated_df[consolidated_df['method'] == method]
                    if not method_data.empty and not pd.isna(method_data.iloc[0]['dbcv_score']):
                        best_score = method_data.iloc[0]['dbcv_score']
                        print(f"{method}: DBCV = {best_score:.4f}")
                    else:
                        print(f"{method}: DBCV = No disponible")
                        
        except Exception as e:
            print(f"Error creando archivo consolidado DBCV: {str(e)}")

    def _save_time(self, metodo, tiempo, modelos_dir):
        os.makedirs(modelos_dir, exist_ok=True)
        csv_path = os.path.join(modelos_dir, "times_clustering.csv")
        df = pd.DataFrame([{
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metodo": metodo,
            "tiempo_segundos": round(tiempo, 2)
        }])
        if os.path.exists(csv_path):
            df_ant = pd.read_csv(csv_path)
            df = pd.concat([df_ant, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
    def _generate_clustering_visualization(self, embeddings, best_params, metodo, modelos_dir, umap_model=None, hdbscan_model=None):
        """
        Genera una visualización del clustering usando los mejores parámetros encontrados.
        
        Parámetros:
            embeddings (np.ndarray): Embeddings originales.
            best_params (dict): Mejores parámetros encontrados.
            metodo (str): Método usado (random, bayesian, grid).
            modelos_dir (str): Directorio donde guardar la imagen.
            umap_model (UMAP, opcional): Modelo UMAP ya entrenado.
            hdbscan_model (HDBSCAN, opcional): Modelo HDBSCAN ya entrenado.
        """
        try:
            # Si los modelos ya están entrenados, usarlos; sino, entrenarlos
            if umap_model is not None and hdbscan_model is not None:
                umap_embeddings = umap_model.transform(embeddings)
                cluster_labels = hdbscan_model.labels_
            else:
                # Aplicar UMAP con los mejores parámetros
                umap_model = umap.UMAP(
                    n_neighbors=best_params['n_neighbors'],
                    n_components=best_params['n_components'],
                    metric='cosine',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                umap_embeddings = umap_model.fit_transform(embeddings)
                
                # Aplicar HDBSCAN
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=best_params['min_cluster_size'],
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                cluster_labels = hdbscan_model.fit_predict(umap_embeddings)
            
            # Crear la visualización
            plt.figure(figsize=(12, 8))
            
            # Usar una paleta de colores adecuada
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters > 1:
                # Si hay ruido (-1), usar una paleta que maneje esto
                if -1 in unique_labels:
                    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters-1))
                    # Añadir gris para el ruido
                    colors = np.vstack([colors, [0.5, 0.5, 0.5, 1.0]])
                    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                                        c=cluster_labels, cmap='Set1', alpha=0.7, s=50)
                else:
                    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                                        c=cluster_labels, cmap='Set1', alpha=0.7, s=50)
            else:
                # Solo un cluster
                scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                                    c='blue', alpha=0.7, s=50)
            
            plt.title(f'Clustering Visualization - {metodo.title()} Search\n'
                     f'Clusters: {n_clusters}, n_neighbors: {best_params["n_neighbors"]}, '
                     f'n_components: {best_params["n_components"]}, '
                     f'min_cluster_size: {best_params["min_cluster_size"]}',
                     fontsize=14, pad=20)
            
            plt.xlabel('UMAP Component 1', fontsize=12)
            plt.ylabel('UMAP Component 2', fontsize=12)
            
            # Añadir colorbar si hay múltiples clusters
            if n_clusters > 1:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Cluster Label', fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar la imagen
            os.makedirs(modelos_dir, exist_ok=True)
            image_path = os.path.join(modelos_dir, f"clustering_visualization_{metodo}.png")
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
            plt.close()  # Cerrar la figura para liberar memoria
            
            print(f"Visualización del clustering guardada en {image_path}")
            
        except Exception as e:
            print(f"Error al generar la visualización del clustering: {str(e)}")
        
    def generate_clusters(self, embeddings, n_neighbors, n_components, min_cluster_size, min_samples=10, min_dist=0.0):
        """
        Aplica UMAP para reducción de dimensionalidad y HDBSCAN para clustering.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            n_neighbors (int): Número de vecinos para UMAP.
            n_components (int): Número de componentes para UMAP.
            min_cluster_size (int): Tamaño mínimo de clúster para HDBSCAN.
            min_samples (int): Número mínimo de muestras para HDBSCAN.
            min_dist (float): Distancia mínima para UMAP.

        Retorna:
            clusters (HDBSCAN object): Objeto ajustado de HDBSCAN con etiquetas y probabilidades.
        """
        umap_embeddings = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric='cosine',
            random_state=self.random_state,
            n_jobs=-1
        ).fit_transform(embeddings)

        clusters = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        ).fit(umap_embeddings)
        return clusters

    def score_clusters(self, clusters, prob_threshold=0.05):
        """
        Evalúa los clusters generados por HDBSCAN.

        Parámetros:
            clusters (HDBSCAN object): Objeto ajustado de HDBSCAN.
            prob_threshold (float): Umbral de probabilidad para considerar un punto como incierto.

        Retorna:
            label_count (int): Número de etiquetas únicas (clusters formados).
            cost (float): Proporción de puntos con baja probabilidad de pertenencia.
        """
        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)
        return label_count, cost

    def random_search(self, embeddings, space, num_evals, save_models=False, save_model=True, modelos_dir="../test/ModeloRamdom"):
        """
        Realiza una búsqueda aleatoria de hiperparámetros para UMAP y HDBSCAN,
        evaluando múltiples combinaciones y registrando sus métricas de agrupamiento.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            space (dict): Espacio de búsqueda con listas de posibles valores para cada hiperparámetro.
            num_evals (int): Número de combinaciones aleatorias a evaluar.

        Retorna:
            result_df (pd.DataFrame): Tabla ordenada por costo, con las combinaciones evaluadas y sus métricas.
        """
        start_time = time.time()
        # Generar combinaciones aleatorias de hiperparámetros
        results = []
        for i in trange(num_evals):
            # Valores UMAP
            n_neighbors = np.random.choice(space['n_neighbors'])
            n_components = np.random.choice(space['n_components'])
            min_dist = np.random.choice(space['min_dist']) if 'min_dist' in space else 0.0
            # Valores HDBSCAN
            min_cluster_size = np.random.choice(space['min_cluster_size'])
            min_samples = np.random.choice(space['min_samples']) if 'min_samples' in space else 10
            # Generar clusters
            clusters = self.generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size, min_samples, min_dist)
            # Evaluar clusters
            label_count, cost = self.score_clusters(clusters)
            results.append([i, n_neighbors, n_components, min_cluster_size, min_samples, min_dist, label_count, cost])
        result_df = pd.DataFrame(
            results,
            columns=['run_id', 'n_neighbors', 'n_components', 'min_cluster_size', 'min_samples', 'min_dist', 'label_count', 'cost']
        ).sort_values(by="cost", ascending=True)

        # Obtener la mejor combinación de hiperparámetros
        best_row = result_df.iloc[0]
        best_params = {
            "n_neighbors": int(best_row["n_neighbors"]),
            "n_components": int(best_row["n_components"]),
            "min_cluster_size": int(best_row["min_cluster_size"]),
            "min_samples": int(best_row["min_samples"]),
            "min_dist": float(best_row["min_dist"])
        }

        # Entrenar modelos UMAP y HDBSCAN con los mejores parámetros y guardar
        umap_model = None
        hdbscan_model = None
        umap_embeddings = None
        
        # Calcular DBCV para los mejores parámetros
        best_clusters = self.generate_clusters(
            embeddings, 
            best_params['n_neighbors'], 
            best_params['n_components'], 
            best_params['min_cluster_size'], 
            best_params['min_samples'],
            best_params['min_dist']
        )
        
        # Obtener embeddings UMAP para DBCV
        temp_umap = umap.UMAP(
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            min_dist=best_params['min_dist'],
            metric='cosine',
            random_state=self.random_state,
            n_jobs=-1
        ).fit_transform(embeddings)
        
        dbcv_score = self._calculate_dbcv(temp_umap, best_clusters.labels_)
        print(f"DBCV Score para mejores parámetros: {dbcv_score}")
        
        # Calcular estadísticas para guardar
        unique_labels = np.unique(best_clusters.labels_)
        n_clusters = len(unique_labels[unique_labels >= 0])
        noise_ratio = np.sum(best_clusters.labels_ == -1) / len(best_clusters.labels_)
        
        # Guardar resultados DBCV en CSV
        self._save_dbcv_results('random_search', best_params, dbcv_score, n_clusters, noise_ratio, modelos_dir)
        
        # Agregar DBCV al DataFrame de resultados
        result_df['dbcv_score'] = None
        result_df.iloc[0, result_df.columns.get_loc('dbcv_score')] = dbcv_score
        
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                min_dist=best_params['min_dist'],
                metric='cosine',
                random_state=self.random_state,
                n_jobs=-1
            ).fit(embeddings)
            umap_embeddings = umap_model.transform(embeddings)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                min_samples=best_params['min_samples'],
                cluster_selection_epsilon=0.0,
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            with open(os.path.join(modelos_dir, "umap_random.pkl"), "wb") as f:
                pickle.dump(umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_random.pkl"), "wb") as f:
                pickle.dump(hdbscan_model, f)
            print(f"Modelos UMAP y HDBSCAN guardados en {modelos_dir}")
            
            # Guardar embeddings originales etiquetados en un CSV
            embeddings_originales_labeled = pd.DataFrame(embeddings)
            embeddings_originales_labeled['label'] = hdbscan_model.labels_
            csv_originales_path = os.path.join(modelos_dir, "embeddings_originales_labeled_random.csv")
            embeddings_originales_labeled.to_csv(csv_originales_path, index=False)
            print(f"Embeddings originales etiquetados guardados en {csv_originales_path}")
            
            # Guardar embeddings originales etiquetados en un NPY
            npy_originales_path = os.path.join(modelos_dir, "embeddings_originales_labeled_random.npy")
            np.save(npy_originales_path, embeddings_originales_labeled.values)
            print(f"Embeddings originales etiquetados guardados en {npy_originales_path}")
            
            # Guardar embeddings UMAP etiquetados en un CSV
            embeddings_umap_labeled = pd.DataFrame(umap_embeddings)
            embeddings_umap_labeled['label'] = hdbscan_model.labels_
            csv_umap_path = os.path.join(modelos_dir, "embeddings_umap_labeled_random.csv")
            embeddings_umap_labeled.to_csv(csv_umap_path, index=False)
            print(f"Embeddings UMAP etiquetados guardados en {csv_umap_path}")
            
            # Guardar embeddings UMAP etiquetados en un NPY
            npy_umap_path = os.path.join(modelos_dir, "embeddings_umap_labeled_random.npy")
            np.save(npy_umap_path, embeddings_umap_labeled.values)
            print(f"Embeddings UMAP etiquetados guardados en {npy_umap_path}")
            
        tiempo = time.time() - start_time
        print(f"Tiempo random_search: {tiempo:.2f} segundos")
        self._save_time("random_search", tiempo, modelos_dir)
        
        # Generar visualización del clustering (reutilizando modelos si están disponibles)
        self._generate_clustering_visualization(embeddings, best_params, "random", modelos_dir, umap_model, hdbscan_model)
        
        # Crear archivo consolidado de resultados DBCV
        self._save_consolidated_dbcv_results(modelos_dir)
        
        return result_df, best_params
            # return result_df.sort_values(by='cost')
    
    #  ________________________________________________________________
    # Bayesian hyperparameter
    # ________________________________________________________________


    def objective(self, params, embeddings, label_lower, label_upper):
        """
        Objective function for hyperopt to minimize, which incorporates constraints
        on the number of clusters we want to identify
        """

        clusters = self.generate_clusters(embeddings,
                                    n_neighbors=params['n_neighbors'],
                                    n_components=params['n_components'],
                                    min_cluster_size=params['min_cluster_size'],
                                    min_samples=params.get('min_samples', 10),
                                    min_dist=params.get('min_dist', 0.0))

        label_count, cost = self.score_clusters(clusters, prob_threshold=0.05)

        #15% penalty on the cost function if outside the desired range of groups
        if (label_count < label_lower) | (label_count > label_upper):
            penalty = 0.15 
        else:
            penalty = 0
        
        loss = cost + penalty
        
        return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}
    

    def bayesian_search(self, embeddings, space, label_lower, label_upper, max_evals=100, csv_path=None, save_models=True, modelos_dir="../test/Modelos"):
        """
        Realiza búsqueda bayesiana sobre el espacio de hiperparámetros usando hyperopt,
        minimizando la función objetivo y guardando los resultados de cada iteración en un CSV.
        Además, guarda los modelos UMAP y HDBSCAN entrenados con los mejores hiperparámetros.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            space (dict): Espacio de búsqueda de hiperparámetros.
            label_lower (int): Límite inferior de clusters deseados.
            label_upper (int): Límite superior de clusters deseados.
            max_evals (int): Número máximo de evaluaciones.
            csv_path (str): Ruta para guardar el CSV de resultados (opcional).
            save_models (bool): Si es True, guarda los modelos entrenados.
            modelos_dir (str): Carpeta donde se guardarán los modelos.

        Retorna:
            best_params (dict): Mejor combinación de hiperparámetros.
            best_clusters (HDBSCAN object): Modelo de clustering con mejores hiperparámetros.
            trials (Trials): Objeto Trials de hyperopt con el historial de evaluaciones.
        """
        start_time = time.time()
        trials = Trials()
        fmin_objective = partial(self.objective, 
                                embeddings=embeddings, 
                                label_lower=label_lower, 
                                label_upper=label_upper)
        best = fmin(fmin_objective, 
                    space=space, 
                    algo=tpe.suggest,
                    max_evals=max_evals, 
                    trials=trials)

        best_params = space_eval(space, best)
        print('best:')
        print(best_params)
        print(f"label count: {trials.best_trial['result']['label_count']}")

        # Guardar resultados de todas las iteraciones en un DataFrame ordenado por 'loss'
        results = []
        for trial in trials.trials:
            vals = trial['misc']['vals']
            result = trial['result']
            params = {k: v[0] if isinstance(v, list) and v else v for k, v in vals.items()}
            params.update({
                'loss': result['loss'],
                'label_count': result['label_count'],
                'status': result['status']
            })
            results.append(params)
        results_df = pd.DataFrame(results).sort_values(by='loss')

        # Calcular DBCV para los mejores parámetros
        best_clusters_for_dbcv = self.generate_clusters(
            embeddings,
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params.get('min_samples', 10)
        )
        
        # Obtener embeddings UMAP para DBCV
        temp_umap_bayes = umap.UMAP(
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            metric='cosine',
            min_dist=best_params.get('min_dist', 0.0),
            random_state=self.random_state, 
            n_jobs=-1
        ).fit_transform(embeddings)
        
        dbcv_score = self._calculate_dbcv(temp_umap_bayes, best_clusters_for_dbcv.labels_)
        print(f"DBCV Score para mejores parámetros bayesianos: {dbcv_score}")
        
        # Calcular estadísticas para guardar
        unique_labels = np.unique(best_clusters_for_dbcv.labels_)
        n_clusters = len(unique_labels[unique_labels >= 0])
        noise_ratio = np.sum(best_clusters_for_dbcv.labels_ == -1) / len(best_clusters_for_dbcv.labels_)
        
        # Guardar resultados DBCV en CSV
        self._save_dbcv_results('bayesian_search', best_params, dbcv_score, n_clusters, noise_ratio, modelos_dir)
        
        # Agregar DBCV al DataFrame de resultados (solo para el mejor resultado)
        results_df['dbcv_score'] = None
        results_df.iloc[0, results_df.columns.get_loc('dbcv_score')] = dbcv_score

        if csv_path:
            results_df.to_csv(csv_path, index=False)
            print(f"Resultados de la búsqueda bayesiana guardados en {csv_path}")

        # Entrenar y guardar modelos UMAP y HDBSCAN con los mejores hiperparámetros
        umap_model = None
        hdbscan_model = None
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                metric='cosine',
                min_dist=best_params.get('min_dist', 0.0),
                random_state=self.random_state, 
                n_jobs=-1
            ).fit(embeddings)
            umap_embeddings = umap_model.transform(embeddings)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                metric='euclidean',
                cluster_selection_method='eom',
                min_samples=best_params.get('min_samples', 10),
                cluster_selection_epsilon=0.0
            ).fit(umap_embeddings)
            with open(os.path.join(modelos_dir, "umap_bayesian.pkl"), "wb") as f:
                pickle.dump(umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_bayesian.pkl"), "wb") as f:
                pickle.dump(hdbscan_model, f)
            print(f"Modelos UMAP y HDBSCAN guardados en {modelos_dir}")
            
            # Guardar embeddings originales etiquetados en un CSV
            embeddings_originales_labeled = pd.DataFrame(embeddings)
            embeddings_originales_labeled['label'] = hdbscan_model.labels_
            csv_originales_path = os.path.join(modelos_dir, "embeddings_originales_labeled_bayesian.csv")
            embeddings_originales_labeled.to_csv(csv_originales_path, index=False)
            print(f"Embeddings originales etiquetados guardados en {csv_originales_path}")
            
            # Guardar embeddings originales etiquetados en un NPY
            npy_originales_path = os.path.join(modelos_dir, "embeddings_originales_labeled_bayesian.npy")
            np.save(npy_originales_path, embeddings_originales_labeled.values)
            print(f"Embeddings originales etiquetados guardados en {npy_originales_path}")
            
            # Guardar embeddings UMAP etiquetados en un CSV
            embeddings_umap_labeled = pd.DataFrame(umap_embeddings)
            embeddings_umap_labeled['label'] = hdbscan_model.labels_
            csv_umap_path = os.path.join(modelos_dir, "embeddings_umap_labeled_bayesian.csv")
            embeddings_umap_labeled.to_csv(csv_umap_path, index=False)
            print(f"Embeddings UMAP etiquetados guardados en {csv_umap_path}")
            
            # Guardar embeddings UMAP etiquetados en un NPY
            npy_umap_path = os.path.join(modelos_dir, "embeddings_umap_labeled_bayesian.npy")
            np.save(npy_umap_path, embeddings_umap_labeled.values)
            print(f"Embeddings UMAP etiquetados guardados en {npy_umap_path}")
            
            
        best_clusters = self.generate_clusters(
            embeddings,
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params.get('min_samples', 10)
        )
        
        tiempo = time.time() - start_time
        print(f"Tiempo bayesian_search: {tiempo:.2f} segundos")
        self._save_time("bayesian_search", tiempo, modelos_dir)
        
        # Generar visualización del clustering (reutilizando modelos si están disponibles)
        self._generate_clustering_visualization(embeddings, best_params, "bayesian", modelos_dir, umap_model, hdbscan_model)

        # Crear archivo consolidado de resultados DBCV
        self._save_consolidated_dbcv_results(modelos_dir)

        return best_params, best_clusters, trials
    
    def separate_grid_search(self, embeddings, umap_space=None, hdbscan_space=None, save_models=True, modelos_dir="../test/Modelos"):
        """
        Realiza GridSearchCV separado: optimiza UMAP primero, luego HDBSCAN.
        Esto es más eficiente y preciso que optimizar ambos juntos.
        
        Parámetros:
            embeddings (np.ndarray): Embeddings a procesar.
            umap_space (dict): Espacio de búsqueda para UMAP (opcional).
            hdbscan_space (dict): Espacio de búsqueda para HDBSCAN (opcional).
            save_models (bool): Si guardar los modelos optimizados.
            modelos_dir (str): Directorio donde guardar resultados.
            
        Retorna:
            results (dict): Diccionario con todos los resultados.
        """
        start_time = time.time()
        
        # Espacios de búsqueda por defecto
        if umap_space is None:
            umap_space = {
                'n_neighbors': [5, 10, 15, 20, 30],
                'n_components': [2, 3, 5, 10],
                'min_dist': [0.0, 0.1, 0.25, 0.5]
            }
        
        if hdbscan_space is None:
            hdbscan_space = {
                'min_cluster_size': [5, 10, 15, 20, 25, 30],
                'min_samples': [1, 5, 10, 15, 20],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3]
            }
        
        print("PASO 1: Optimizando UMAP...")
        
        # GridSearch para UMAP
        umap_estimator = UMAPEstimator(random_state=self.random_state)
        umap_grid = GridSearchCV(
            estimator=umap_estimator,
            param_grid=umap_space,
            cv=3,
            n_jobs=-1,
            verbose=3
        )
        
        umap_grid.fit(embeddings)
        best_umap_params = umap_grid.best_params_
        best_umap_model = umap_grid.best_estimator_
        
        
        print(f"Mejores parámetros UMAP: {best_umap_params}")
        
        print("PASO 2: Generando embeddings UMAP...")
        umap_embeddings = best_umap_model.transform(embeddings)
        
        print("PASO 3: Optimizando HDBSCAN...")
        
        # GridSearch para HDBSCAN
        hdbscan_estimator = HDBSCANEstimator()
        hdbscan_grid = GridSearchCV(
            estimator=hdbscan_estimator,
            param_grid=hdbscan_space,
            cv=3,
            n_jobs=-1,
            verbose=3
        )
        
        hdbscan_grid.fit(umap_embeddings)
        best_hdbscan_params = hdbscan_grid.best_params_
        best_hdbscan_model = hdbscan_grid.best_estimator_
        
        print(f"Mejores parámetros HDBSCAN: {best_hdbscan_params}")
        
        # Estadísticas finales
        final_labels = best_hdbscan_model.labels_
        unique_labels = np.unique(final_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(final_labels == -1)
        
        print(f"Clusters encontrados: {n_clusters}")
        print(f"Puntos de ruido: {n_noise} ({n_noise/len(final_labels)*100:.1f}%)")
        
        # Combinar mejores parámetros para compatibilidad
        combined_best_params = {**best_umap_params, **best_hdbscan_params}
        
        # Calcular DBCV con los mejores parámetros
        dbcv_score = self._calculate_dbcv(umap_embeddings, final_labels)
        print(f"DBCV Score para Grid Search: {dbcv_score}")
        
        # Guardar resultados DBCV en CSV
        self._save_dbcv_results('separate_grid_search', combined_best_params, dbcv_score, n_clusters, n_noise / len(final_labels), modelos_dir)
        
        # Guardar modelos si se solicita
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            
            # Guardar modelos individuales
            with open(os.path.join(modelos_dir, "umap_separate_grid.pkl"), "wb") as f:
                pickle.dump(best_umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_separate_grid.pkl"), "wb") as f:
                pickle.dump(best_hdbscan_model, f)
            
            # Guardar embeddings y resultados
            np.save(os.path.join(modelos_dir, "umap_embeddings_separate_grid.npy"), umap_embeddings)
            
            # Guardar embeddings originales etiquetados
            embeddings_labeled = pd.DataFrame(embeddings)
            embeddings_labeled['label'] = final_labels
            embeddings_labeled.to_csv(os.path.join(modelos_dir, "embeddings_originales_labeled_separate_grid.csv"), index=False)
            
            # Guardar embeddings UMAP etiquetados
            umap_labeled = pd.DataFrame(umap_embeddings)
            umap_labeled['label'] = final_labels
            umap_labeled.to_csv(os.path.join(modelos_dir, "embeddings_umap_labeled_separate_grid.csv"), index=False)
            
            print(f"Modelos y resultados guardados en {modelos_dir}")
        
        tiempo_total = time.time() - start_time
        print(f"Tiempo total: {tiempo_total:.2f} segundos")
        self._save_time("separate_grid_search", tiempo_total, modelos_dir)
        
        # Generar visualización
        self._generate_clustering_visualization(embeddings, combined_best_params, "separate_grid", modelos_dir, 
                                               best_umap_model.umap_model_, best_hdbscan_model.hdbscan_model_)
        
        # Crear archivo consolidado de resultados DBCV
        self._save_consolidated_dbcv_results(modelos_dir)
        
        # Crear DataFrame de resultados para compatibilidad
        result_df = pd.DataFrame([{
            'run_id': 0,
            'n_neighbors': combined_best_params['n_neighbors'],
            'n_components': combined_best_params['n_components'],
            'min_cluster_size': combined_best_params['min_cluster_size'],
            'min_samples': combined_best_params['min_samples'],
            'label_count': n_clusters,
            'cost': n_noise / len(final_labels),  # Usar proporción de ruido como costo
            'dbcv_score': dbcv_score
        }])
        
        return result_df, combined_best_params
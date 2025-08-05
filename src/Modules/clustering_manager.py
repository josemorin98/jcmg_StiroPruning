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
        
    def generate_clusters(self, embeddings, n_neighbors, n_components, min_cluster_size, min_samples=10):
        """
        Aplica UMAP para reducción de dimensionalidad y HDBSCAN para clustering.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            n_neighbors (int): Número de vecinos para UMAP.
            n_components (int): Número de componentes para UMAP.
            min_cluster_size (int): Tamaño mínimo de clúster para HDBSCAN.
            min_samples (int): Número mínimo de muestras para HDBSCAN.

        Retorna:
            clusters (HDBSCAN object): Objeto ajustado de HDBSCAN con etiquetas y probabilidades.
        """
        umap_embeddings = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
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
            # Valores HDBSCAN
            min_cluster_size = np.random.choice(space['min_cluster_size'])
            min_samples = np.random.choice(space['min_samples']) if 'min_samples' in space else 10
            # Generar clusters
            clusters = self.generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size, min_samples)
            # Evaluar clusters
            label_count, cost = self.score_clusters(clusters)
            results.append([i, n_neighbors, n_components, min_cluster_size, min_samples, label_count, cost])
        result_df = pd.DataFrame(
            results,
            columns=['run_id', 'n_neighbors', 'n_components', 'min_cluster_size', 'min_samples', 'label_count', 'cost']
        ).sort_values(by="cost", ascending=True)

        # Obtener la mejor combinación de hiperparámetros
        best_row = result_df.iloc[0]
        best_params = {
            "n_neighbors": int(best_row["n_neighbors"]),
            "n_components": int(best_row["n_components"]),
            "min_cluster_size": int(best_row["min_cluster_size"]),
            "min_samples": int(best_row["min_samples"])
        }

        # Entrenar modelos UMAP y HDBSCAN con los mejores parámetros y guardar
        umap_model = None
        hdbscan_model = None
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                min_dist=0.0,
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
                                    min_samples=params.get('min_samples', 10))

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
                min_dist=0.0,
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
            n_jobs=1,
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
            n_jobs=1,
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
        
        # Crear DataFrame de resultados para compatibilidad
        result_df = pd.DataFrame([{
            'run_id': 0,
            'n_neighbors': combined_best_params['n_neighbors'],
            'n_components': combined_best_params['n_components'],
            'min_cluster_size': combined_best_params['min_cluster_size'],
            'min_samples': combined_best_params['min_samples'],
            'label_count': n_clusters,
            'cost': n_noise / len(final_labels)  # Usar proporción de ruido como costo
        }])
        
        return result_df, combined_best_params
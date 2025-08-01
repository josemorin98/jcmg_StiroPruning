import umap
import hdbscan
import numpy as np
import pandas as pd
from tqdm import trange
from functools import partial
from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK
import os 
import pickle
import time

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
        
    def generate_clusters(self, embeddings, n_neighbors, n_components, min_cluster_size):
        """
        Aplica UMAP para reducción de dimensionalidad y HDBSCAN para clustering.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            n_neighbors (int): Número de vecinos para UMAP.
            n_components (int): Número de componentes para UMAP.
            min_cluster_size (int): Tamaño mínimo de clúster para HDBSCAN.

        Retorna:
            clusters (HDBSCAN object): Objeto ajustado de HDBSCAN con etiquetas y probabilidades.
        """
        umap_embeddings = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            # random_state=self.random_state,

            n_jobs=-1
        ).fit_transform(embeddings)

        clusters = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
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
            n_neighbors = np.random.choice(space['n_neighbors'])
            n_components = np.random.choice(space['n_components'])
            min_cluster_size = np.random.choice(space['min_cluster_size'])
            clusters = self.generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size)
            label_count, cost = self.score_clusters(clusters)
            results.append([i, n_neighbors, n_components, min_cluster_size, label_count, cost])
        result_df = pd.DataFrame(
            results,
            columns=['run_id', 'n_neighbors', 'n_components', 'min_cluster_size', 'label_count', 'cost']
        )
        
         # Obtener la mejor combinación de hiperparámetros
        best_row = result_df.iloc[0]
        best_params = {
            "n_neighbors": int(best_row["n_neighbors"]),
            "n_components": int(best_row["n_components"]),
            "min_cluster_size": int(best_row["min_cluster_size"])
        }

        # Entrenar modelos UMAP y HDBSCAN con los mejores parámetros y guardar
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                metric='cosine',
                # random_state=self.random_state,
                n_jobs=-1
            ).fit(embeddings)
            umap_embeddings = umap_model.transform(embeddings)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            with open(os.path.join(modelos_dir, "umap_random.pkl"), "wb") as f:
                pickle.dump(umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_random.pkl"), "wb") as f:
                pickle.dump(hdbscan_model, f)
            print(f"Modelos UMAP y HDBSCAN guardados en {modelos_dir}")
            
            # Guardar embeddings etiquetados en un CSV
            embeddings_labeled = pd.DataFrame(umap_embeddings)
            embeddings_labeled['label'] = hdbscan_model.labels_
            csv_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_random.csv")
            embeddings_labeled.to_csv(csv_labeled_path, index=False)
            print(f"Embeddings etiquetados guardados en {csv_labeled_path}")
            
            # Guardar embeddings etiquetados en un NPY
            npy_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_random.npy")
            np.save(npy_labeled_path, embeddings_labeled.values)
            print(f"Embeddings etiquetados guardados en {npy_labeled_path}")
            
        tiempo = time.time() - start_time
        print(f"Tiempo random_search: {tiempo:.2f} segundos")
        self._save_time("random_search", tiempo, modelos_dir)
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
                                    min_cluster_size=params['min_cluster_size'])

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
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                metric='cosine',
                # random_state=best_params.get('random_state', self.random_state),
                n_jobs=-1
            ).fit(embeddings)
            umap_embeddings = umap_model.transform(embeddings)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            with open(os.path.join(modelos_dir, "umap_bayesian.pkl"), "wb") as f:
                pickle.dump(umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_bayesian.pkl"), "wb") as f:
                pickle.dump(hdbscan_model, f)
            print(f"Modelos UMAP y HDBSCAN guardados en {modelos_dir}")
            # Guardar embeddings etiquetados en un CSV
            embeddings_labeled = pd.DataFrame(umap_embeddings)
            embeddings_labeled['label'] = hdbscan_model.labels_
            csv_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_bayesian.csv")
            embeddings_labeled.to_csv(csv_labeled_path, index=False)
            print(f"Embeddings etiquetados guardados en {csv_labeled_path}")
            
            # Guardar embeddings etiquetados en un NPY
            npy_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_bayesian.npy")
            np.save(npy_labeled_path, embeddings_labeled.values)
            print(f"Embeddings etiquetados guardados en {npy_labeled_path}")
            
            
        best_clusters = self.generate_clusters(
            embeddings,
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            min_cluster_size=best_params['min_cluster_size'],
            # random_state=best_params.get('random_state', self.random_state)
        )
        
        tiempo = time.time() - start_time
        print(f"Tiempo bayesian_search: {tiempo:.2f} segundos")
        self._save_time("bayesian_search", tiempo, modelos_dir)
    

        return best_params, best_clusters, trials
    
    def grid_search(self, embeddings, space, save_models=True, modelos_dir="../test/Modelos"):
        """
        Realiza una búsqueda exhaustiva (grid search) de hiper
        parámetros para UMAP y HDBSCAN,
        evaluando todas las combinaciones posibles y registrando sus métricas de agrupamiento.
        Guarda los mejores modelos UMAP y HDBSCAN si save_models=True.

        Parámetros:
            embeddings (np.ndarray): Embeddings a reducir y agrupar.
            space (dict): Diccionario con listas de valores para cada hiperparámetro.
            save_models (bool): Si es True, guarda los modelos entrenados con la mejor combinación.
            modelos_dir (str): Carpeta donde se guardarán los modelos.

        Retorna:
            result_df (pd.DataFrame): Tabla ordenada por costo, con las combinaciones evaluadas y sus métricas.
            best_params (dict): Mejor combinación de hiperparámetros.
        """
        from itertools import product
        start_time = time.time()
        # Generar todas las combinaciones de hiperparámetros
        results = []
        param_grid = list(product(
            space['n_neighbors'],
            space['n_components'],
            space['min_cluster_size']
        ))

        for i, (n_neighbors, n_components, min_cluster_size) in enumerate(param_grid):
            clusters = self.generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size)
            label_count, cost = self.score_clusters(clusters)
            results.append([i, n_neighbors, n_components, min_cluster_size, label_count, cost])

        result_df = pd.DataFrame(
            results,
            columns=['run_id', 'n_neighbors', 'n_components', 'min_cluster_size', 'label_count', 'cost']
        ).sort_values(by='cost')

        # Obtener la mejor combinación de hiperparámetros
        best_row = result_df.iloc[0]
        best_params = {
            "n_neighbors": int(best_row["n_neighbors"]),
            "n_components": int(best_row["n_components"]),
            "min_cluster_size": int(best_row["min_cluster_size"])
        }

        # Entrenar y guardar modelos UMAP y HDBSCAN con los mejores parámetros
        if save_models:
            os.makedirs(modelos_dir, exist_ok=True)
            umap_model = umap.UMAP(
                n_neighbors=best_params['n_neighbors'],
                n_components=best_params['n_components'],
                metric='cosine',
                n_jobs=-1
            ).fit(embeddings)
            umap_embeddings = umap_model.transform(embeddings)
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit(umap_embeddings)
            with open(os.path.join(modelos_dir, "umap_grid.pkl"), "wb") as f:
                pickle.dump(umap_model, f)
            with open(os.path.join(modelos_dir, "hdbscan_grid.pkl"), "wb") as f:
                pickle.dump(hdbscan_model, f)
            print(f"Modelos UMAP y HDBSCAN guardados en {modelos_dir} (grid search)")
            
            # Guardar embeddings etiquetados en un CSV
            embeddings_labeled = pd.DataFrame(umap_embeddings)
            embeddings_labeled['label'] = hdbscan_model.labels_
            csv_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_grid.csv")
            embeddings_labeled.to_csv(csv_labeled_path, index=False)
            print(f"Embeddings etiquetados guardados en {csv_labeled_path}")
            
            # Guardar embeddings etiquetados en un NPY
            npy_labeled_path = os.path.join(modelos_dir, "embeddings_labeled_grid.npy")
            np.save(npy_labeled_path, embeddings_labeled.values)
            print(f"Embeddings etiquetados guardados en {npy_labeled_path}")
        tiempo = time.time() - start_time
        print(f"Tiempo grid_search: {tiempo:.2f} segundos")
        self._save_time("grid_search", tiempo, modelos_dir)
    
        return result_df, best_params
import pandas as pd
import numpy as np
import time
import os
import argparse
from Modules.clustering_manager import ClusteringManager
from Modules.grid_search import GridSearchManager
from hyperopt import hp

def main():
    
    parser = argparse.ArgumentParser(description="Buscar hiperparámetros óptimos para clustering.")
    parser.add_argument(
        "--modelo",
        type=str,
        choices=["use", "st1", "st2", "st3"],
        required=True,
        help="Modelo de embeddings a utilizar: use, st1, st2 o st3"
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=10,
        help="Número de evaluaciones para cada búsqueda"
    )
    args = parser.parse_args()

    path_test = "../test"
    path_embeddings = f"{path_test}/embeddings/{args.modelo}/{args.modelo}_1753994950.252781.npy"
    assert os.path.exists(path_embeddings), f"Embeddings no encontrados en {path_embeddings}"

    print(f"Cargando embeddings desde {path_embeddings} ...")
    embeddings = np.load(path_embeddings)

    clustering = ClusteringManager(random_state=42, model=args.modelo)

    # ------------------- Random Search -------------------
    print("\nIniciando búsqueda aleatoria de hiperparámetros...")
    start_time = time.time()
    configSpace_random = {
        "n_neighbors": range(12, 16),
        "n_components": range(3, 7),
        "min_cluster_size": range(2, 16)
    }
    random_results, best_random_params = clustering.random_search(
        embeddings=embeddings,
        space=configSpace_random,
        num_evals=args.max_evals,
        save_models=True,
        modelos_dir=f"{path_test}/Modelos_{args.modelo}",
    )
    random_results.to_csv(f"{path_test}/random_{args.modelo}.csv", index=False)
    print(f"Búsqueda aleatoria completada en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros (random): {best_random_params}")

    # ------------------- Bayesian Search -------------------
    print("\nIniciando búsqueda bayesiana de hiperparámetros...")
    start_time = time.time()
    configSpace_bayes = {
        "n_neighbors": hp.choice("n_neighbors", list(range(12, 16))),
        "n_components": hp.choice("n_components", list(range(3, 7))),
        "min_cluster_size": hp.choice("min_cluster_size", list(range(2, 16))),
        "random_state": 42
    }
    label_lower = 2
    label_upper = 100
    bayes_params, bayes_clusters, trials = clustering.bayesian_search(
        embeddings=embeddings,
        space=configSpace_bayes,
        label_lower=label_lower,
        label_upper=label_upper,
        max_evals=args.max_evals,
        csv_path=f"{path_test}/bayesian_{args.modelo}.csv",
        save_models=True,
        modelos_dir=f"{path_test}/Modelos_{args.modelo}",
    )
    print(f"Búsqueda bayesiana completada en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros (bayesiano): {bayes_params}")



    # ------------------- Grid Search -------------------
    # Definir el espacio de búsqueda
    param_grid = {
        "n_neighbors": range(12, 16),
        "n_components": range(3, 7),
        "min_cluster_size": range(2, 16)
    }
    
    # model = GridSearchManager()
    # best_params, best_score, results_df = model.run_grid_search(embeddings, param_grid, name=args.modelo, modelos_dir=f"{path_test}/Modelos")
    
    # print("Mejores parámetros (grid):", best_params)
    # print("Mejor score (grid):", best_score)

if __name__ == "__main__":
    # print(tf.config.list_physical_devices('GPU'))
    main()
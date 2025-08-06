import pandas as pd
import numpy as np
import time
import os
import argparse
from Modules.clustering_manager import ClusteringManager
# from Modules.grid_search import GridSearchManager
from hyperopt import hp

def main():
    
    # -------------------------------------------------------------------------------
    # Argumentos de línea de comandos para seleccionar el modelo y los hiperparámetros
    parser = argparse.ArgumentParser(description="Buscar hiperparámetros óptimos para clustering con embeddings ajustados (spatial, temporal, interest).")
    parser.add_argument("--modelo", type=str, choices=["use", "st1", "st2", "st3"], required=True, help="Modelo de embeddings a utilizar: use, st1, st2 o st3")
    parser.add_argument("--max_evals", type=int, default=10, help="Número de evaluaciones para cada búsqueda")
    parser.add_argument("--label_lower", type=int, default=100, help="Límite inferior para las etiquetas")
    parser.add_argument("--label_upper", type=int, default=1000, help="Límite superior para las etiquetas")
    parser.add_argument("--use_adjusted", action="store_true", help="Usar embeddings ajustados con columnas spatial, temporal e interest")

    # Parse los argumentos
    args = parser.parse_args()
    path_test = "../test"

    # Verificar si el modelo es válido
    if args.modelo not in ["use", "st1", "st2", "st3"]:
        raise ValueError("Modelo no válido. Debe ser 'use', 'st1', 'st2' o 'st3'.")
    
    # Determinar la ruta de embeddings y sufijos según si se usan ajustados o no
    if args.use_adjusted:
        path_embeddings = f"{path_test}/embeddings/{args.modelo}/{args.modelo}.npy"
        output_suffix = "_adjusted"
        print(f"Modelo seleccionado: {args.modelo} (usando embeddings ajustados)")
    else:
        path_embeddings = f"{path_test}/embeddings/{args.modelo}/{args.modelo}.npy"
        output_suffix = ""
        print(f"Modelo seleccionado: {args.modelo} (usando embeddings originales)")
    
    # Verificar si el directorio de embeddings existe
    if not os.path.exists(f"{path_test}/embeddings/{args.modelo}"):
        raise FileNotFoundError(f"El directorio de embeddings para el modelo {args.modelo} no existe.")

    # Verificar que el archivo de embeddings específico existe
    assert os.path.exists(path_embeddings), f"Archivo de embeddings no encontrado en {path_embeddings}"

    print(f"Cargando embeddings desde {path_embeddings} ...")
    embeddings = np.load(path_embeddings)
    print(f"Dimensiones de embeddings: {embeddings.shape}")

    clustering = ClusteringManager(random_state=42, model=args.modelo)

    # Ajustar rangos de hiperparámetros para embeddings con información adicional
    if args.use_adjusted:
        # Parámetros optimizados para embeddings con información espacial/temporal/interés
        configSpace_random = {
            "n_neighbors": np.arange(30, 50, 2),
            "n_components": np.arange(3, 4),
            "min_cluster_size": np.arange(120, 200),
            "min_samples": np.arange(50, 150),
            "min_dist": [0.0, 0.1, 0.25]
        }
        configSpace_bayes = {
            "n_neighbors": hp.choice("n_neighbors", list(range(30, 50, 2))),
            "n_components": hp.choice("n_components", list(range(3, 4))),
            "min_samples": hp.choice("min_samples", list(range(120, 200))),
            "min_cluster_size": hp.choice("min_cluster_size", list(range(50, 150))),
            "min_dist": hp.choice("min_dist", [0.0, 0.1, 0.25]),
            "random_state": 42
        }

    else:
        # Parámetros originales para embeddings estándar
        configSpace_random = {
            "n_neighbors": np.arange(30, 100),
            "n_components": np.arange(2, 4),
            "min_cluster_size": np.arange(100, 200, 20),
            "min_samples": np.arange(50, 150, 5),
            "min_dist": [0.0, 0.1, 0.25]
        }
        configSpace_bayes = {
            "n_neighbors": hp.choice("n_neighbors", list(range(30, 100))),
            "n_components": hp.choice("n_components", list(range(2, 4))),
            "min_cluster_size": hp.choice("min_cluster_size", list(range(100, 200, 20))),
            "min_samples": hp.choice("min_samples", list(range(50, 150, 5))),
            "min_dist": hp.choice("min_dist", [0.0, 0.1, 0.25]),
            "random_state": 42
        }


    # Variables para almacenar resultados
    random_results, best_random_params = None, None
    bayes_params, bayes_clusters, trials = None, None, None
    grid_results, best_grid_params = None, None
    
    # ===============================================================================
    # EJECUTAR TODAS LAS BÚSQUEDAS
    # ===============================================================================
    
    print(f"\nEjecutando todas las búsquedas de hiperparámetros")
    print(f"Directorio de modelos: {path_test}/Modelos_{args.modelo}{output_suffix}")
    
    # # ------------------- Random Search -------------------
    print("\n" + "="*50)
    print("RANDOM SEARCH")
    print("="*50)
    start_time = time.time()

    random_results, best_random_params = clustering.random_search(
        embeddings=embeddings,
        space=configSpace_random,
        num_evals=args.max_evals,
        save_models=True,
        modelos_dir=f"{path_test}/Modelos_{args.modelo}{output_suffix}",
    )
    random_results.to_csv(f"{path_test}/random_{args.modelo}{output_suffix}.csv", index=False)
    print(f"Random Search completado en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros (random): {best_random_params}")

    # # ------------------- Bayesian Search -------------------
    print("\n" + "="*50)
    print("BAYESIAN SEARCH")
    print("="*50)
    start_time = time.time()
  
    label_lower = args.label_lower
    label_upper = args.label_upper
    bayes_params, bayes_clusters, trials = clustering.bayesian_search(
        embeddings=embeddings,
        space=configSpace_bayes,
        label_lower=label_lower,
        label_upper=label_upper,
        max_evals=args.max_evals,
        csv_path=f"{path_test}/bayesian_{args.modelo}{output_suffix}.csv",
        save_models=True,
        modelos_dir=f"{path_test}/Modelos_{args.modelo}{output_suffix}",
    )
    print(f"Bayesian Search completado en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros (bayesian): {bayes_params}")

    # ------------------- Grid Search -------------------
    print("\n" + "="*50)
    print("GRID SEARCH SEPARADO (UMAP + HDBSCAN)")
    print("="*50)
    start_time = time.time()
    
    # Configurar parámetros separados para UMAP y HDBSCAN
    if args.use_adjusted:
        # Parámetros para embeddings ajustados
        umap_param_grid = {
            # 'n_neighbors': [15, 20, 25, 30, 35, 40, 45],
            # 'n_components': [2, 3],
            "n_neighbors": np.arange(30, 50, 2),
            "n_components": np.arange(3, 4),
            'min_dist': [0.0, 0.1, 0.25]
            # "min_samples": np.arange(50, 150)
        }
        hdbscan_param_grid = {
            'min_cluster_size': np.arange(100,200,20),# [120, 140, 160, 180],
            'min_samples': np.arange(50, 150, 5),
            # 'cluster_selection_epsilon': [0.0, 0.1, 0.2]
        }
    else:
        # Parámetros para embeddings originales
        umap_param_grid = {
            'n_neighbors': [30, 45, 60, 75, 90],
            'n_components': [2, 3],
            'min_dist': [0.0, 0.1, 0.25]
        }
        hdbscan_param_grid = {
            'min_cluster_size': [10, 20, 30, 40],
            'min_samples': [10, 20, 30, 40],
            # 'cluster_selection_epsilon': [0.0, 0.1, 0.2]
        }
    
    # Calcular número total de combinaciones
    umap_combinations = 1
    for values in umap_param_grid.values():
        umap_combinations *= len(values)
    
    hdbscan_combinations = 1
    for values in hdbscan_param_grid.values():
        hdbscan_combinations *= len(values)
    
    print(f"Combinaciones UMAP: {umap_combinations}")
    print(f"Combinaciones HDBSCAN: {hdbscan_combinations}")
    print(f"Total de evaluaciones: {umap_combinations + hdbscan_combinations}")
    
    # Ejecutar Grid Search separado
    grid_results, combined_best_params = clustering.separate_grid_search(
        embeddings=embeddings,
        umap_space=umap_param_grid,
        hdbscan_space=hdbscan_param_grid,
        save_models=True,
        modelos_dir=f"{path_test}/Modelos_{args.modelo}{output_suffix}",
    )
    
    # Depuración: mostrar estructura real del diccionario retornado
    print("combined_best_params:", combined_best_params)

    # Extraer mejores parámetros
    umap_params = {k: combined_best_params[k] for k in ['n_neighbors', 'n_components', 'min_dist'] if k in combined_best_params}
    hdbscan_params = {k: combined_best_params[k] for k in ['min_cluster_size', 'min_samples'] if k in combined_best_params}
    
    # Extraer valores del DataFrame de resultados
    n_clusters = grid_results.iloc[0]['label_count']
    noise_ratio = grid_results.iloc[0]['cost']
    dbcv_score = grid_results.iloc[0]['dbcv_score'] if 'dbcv_score' in grid_results.columns else None
    
    best_grid_params = {
        'umap_params': umap_params,
        'hdbscan_params': hdbscan_params,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'dbcv_score': dbcv_score
    }
    
    # Guardar resultados
    grid_summary = {
        'optimization_method': 'separate_gridsearch',
        'model': args.modelo,
        'use_adjusted': args.use_adjusted,
        **umap_params,
        **hdbscan_params,
        'final_n_clusters': n_clusters,
        'final_noise_ratio': noise_ratio,
        'dbcv_score': dbcv_score,
        'total_time_seconds': time.time() - start_time
    }
    
    pd.DataFrame([grid_summary]).to_csv(f"{path_test}/grid_{args.modelo}{output_suffix}.csv", index=False)
    print(f"Grid Search completado en {time.time() - start_time:.2f} segundos.")
    print(f"Mejores parámetros UMAP: {umap_params}")
    print(f"Mejores parámetros HDBSCAN: {hdbscan_params}")
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Ratio de ruido: {noise_ratio:.3f}")
    if dbcv_score is not None:
        print(f"DBCV Score: {dbcv_score:.3f}")
    else:
        print("DBCV Score: No disponible")

    # ===============================================================================
    # RESUMEN FINAL DE TODAS LAS BÚSQUEDAS
    # ===============================================================================
    print("\n" + "="*80)
    print(f"RESUMEN FINAL - Modelo: {args.modelo} {'(ajustado)' if args.use_adjusted else '(original)'}")
    print("="*80)
    print(f"Dimensiones embeddings: {embeddings.shape}")
    print(f"Evaluaciones por búsqueda: {args.max_evals}")
    
    print(f"\nRandom Search - Mejores parámetros:")
    for key, value in best_random_params.items():
        print(f"   {key}: {value}")
    
    print(f"\nBayesian Search - Mejores parámetros:")
    for key, value in bayes_params.items():
        if key != 'random_state':
            print(f"   {key}: {value}")
    
    print(f"\nGrid Search Separado - Mejores parámetros:")
    print(f"   UMAP:")
    for key, value in best_grid_params['umap_params'].items():
        print(f"      {key}: {value}")
    print(f"   HDBSCAN:")
    for key, value in best_grid_params['hdbscan_params'].items():
        print(f"      {key}: {value}")
    print(f"   Clusters finales: {best_grid_params['n_clusters']}")
    print(f"   Ratio de ruido: {best_grid_params['noise_ratio']:.3f}")
    if best_grid_params['dbcv_score'] is not None:
        print(f"   DBCV Score: {best_grid_params['dbcv_score']:.3f}")
    else:
        print(f"   DBCV Score: No disponible")
    
    print(f"\nArchivos guardados en: {path_test}")
    print(f"Modelos guardados en: {path_test}/Modelos_{args.modelo}{output_suffix}/")
    
    # Mostrar archivos generados
    generated_files = [
        # f"random_{args.modelo}{output_suffix}.csv",
        # f"bayesian_{args.modelo}{output_suffix}.csv",
        f"grid_{args.modelo}{output_suffix}.csv"
    ]


if __name__ == "__main__":
    main()
import tensorflow as tf
import tensorflow_hub as hub
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
import random
import pandas as pd
from tqdm import trange
import umap
import numpy as np
import hdbscan

# Cargar el dataset de intents
print("Cargando dataset...")
data_sample = pd.read_csv("../data/sample.csv")
# all_intents = list(data_sample['text'])

# Concatenan todas las sentencias separadas por un espacio
# Reemplazar espacios por "_" en cada celda y luego concatenar con espacios
print("\nProcesando texto: reemplazando espacios y concatenando columnas...")
all_intents = data_sample.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()

print(f"\nTotal de registros procesados: {len(all_intents)}")
print("Ejemplo de texto procesado:")
print(all_intents[0])

# define the document embedding models to use for comparison
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model_use = hub.load(module_url)
# model_st1 = SentenceTransformer('all-mpnet-base-v2')
# model_st2 = SentenceTransformer('all-MiniLM-L6-v2')
# model_st3 = SentenceTransformer('paraphrase-mpnet-base-v2')
print("Modelo USE cargado.")

# Guardar modelo para embeding posterior al query
# Guardar los 4 modelos para pruebas
# Realizar queries exactos para busqueds de ramas


def embed(model, model_type, sentences):
    """
    wrapper function for generating message embeddings
    """
    print(f"\nGenerando embeddings usando modelo: {model_type}...")
    if (model_type == 'use'):
        embeddings = model(sentences)
    elif model_type == 'sentence transformer':
        embeddings = model.encode(sentences)
    print(f"Embeddings generados: {embeddings.shape}")
    return embeddings

# generate embeddings for each model
embeddings_use = embed(model_use, 'use', all_intents)
# embeddings_st1 = embed(model_st1, 'sentence transformer', all_intents)
# embeddings_st2 = embed(model_st2, 'sentence transformer', all_intents)
# embeddings_st3 = embed(model_st3, 'sentence transformer', all_intents)


# ________________________________________________________________
# Generating UMAP and HDBSCAN
# ________________________________________________________________
print("\n\n\n UMAP and HDBSCAN")

def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components, 
                      min_cluster_size,
                      random_state = 42):
    """
    Genera agrupamientos (clusters) aplicando reducción de dimensionalidad con UMAP 
    y posterior agrupamiento con HDBSCAN.

    Parámetros:
        message_embeddings (array): Representaciones vectoriales de los textos.
        n_neighbors (int): Número de vecinos considerados por UMAP.
        n_components (int): Número de dimensiones resultantes tras aplicar UMAP.
        min_cluster_size (int): Tamaño mínimo permitido para un clúster en HDBSCAN.
        random_state (int, opcional): Semilla aleatoria para reproducibilidad.

    Retorna:
        clusters (HDBSCAN object): Objeto de HDBSCAN con los resultados del agrupamiento.
    """
    
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric='cosine', 
                                random_state=random_state,
                                n_jobs=-1) # trabajadores
                            .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', 
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters

def score_clusters(clusters, prob_threshold = 0.05):
    """
    Evalúa los clusters generados por HDBSCAN, devolviendo la cantidad de etiquetas únicas 
    y un costo basado en la incertidumbre de los puntos asignados.

    Parámetros:
        clusters (HDBSCAN object): Objeto de HDBSCAN que contiene etiquetas y probabilidades.
        prob_threshold (float): Umbral de probabilidad para considerar un punto como incierto.

    Retorna:
        label_count (int): Número total de etiquetas únicas (clusters formados).
        cost (float): Proporción de puntos cuya probabilidad de pertenencia al clúster es menor al umbral.
    """
    
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
    
    return label_count, cost


def random_search(embeddings, space, num_evals):
    """
    Realiza una búsqueda aleatoria de hiperparámetros para UMAP y HDBSCAN,
    evaluando múltiples combinaciones y registrando sus métricas de agrupamiento.

    Parámetros:
        embeddings (array): Representaciones vectoriales a reducir y agrupar.
        space (dict): Espacio de búsqueda con listas de posibles valores para cada hiperparámetro.
        num_evals (int): Número de combinaciones aleatorias a evaluar.

    Retorna:
        result_df (DataFrame): Tabla ordenada por costo, con las combinaciones evaluadas y sus métricas.
    """
    
    results = []
    
    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        
        clusters = generate_clusters(embeddings, 
                                     n_neighbors = n_neighbors, 
                                     n_components = n_components, 
                                     min_cluster_size = min_cluster_size)
    
        label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
                
        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        label_count, cost])
    
    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components', 
                                               'min_cluster_size', 'label_count', 'cost'])
    
    return result_df.sort_values(by='cost')

configSpace = {
    "n_neighbors":range(12,16),
    "n_components":range(3,7),
    "min_cluster_size":range(2,16)
}

random_use = random_search(embeddings=embeddings_use,
                           space=configSpace,
                          num_evals=10)
random_use.to_csv("../test/random_use.csv")

print("\n\n\n Bayesian hyperparameter")
# ________________________________________________________________
# Bayesian hyperparameter
# ________________________________________________________________

from functools import partial
from hyperopt import fmin, tpe, Trials, space_eval, hp, STATUS_OK

def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify
    """
    
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 random_state = params['random_state'])
    
    label_count, cost = score_clusters(clusters, prob_threshold = 0.05)
    
    #15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15 
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayseian search on hyperopt hyperparameter space to minimize objective function
    """
    
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=max_evals, 
                trials=trials)

    best_params = space_eval(space, best)
    print ('best:')
    print (best_params)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    
    best_clusters = generate_clusters(embeddings, 
                                      n_neighbors = best_params['n_neighbors'], 
                                      n_components = best_params['n_components'], 
                                      min_cluster_size = best_params['min_cluster_size'],
                                      random_state = best_params['random_state'])
    
    return best_params, best_clusters, trials

configSpace = {
    "n_neighbors":hp.choice("n_neighbors", range(3,16)),
    "n_components":hp.choice("n_components", range(3,16)),
    "min_cluster_size":hp.choice("min_cluster_size", range(2,16)),
    "random_state":42
}

label_lower = 30
label_upper = 100
max_evals = 10

best_param_use, best_cluster_use, trials_use = bayesian_search(embeddings=embeddings_use,
                                                               space=configSpace,
                                                               label_lower=label_lower,
                                                               label_upper=label_upper,
                                                               max_evals=max_evals)

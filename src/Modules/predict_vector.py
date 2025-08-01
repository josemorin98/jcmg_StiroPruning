import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Modules.model_manager import EmbeddingModelManager

class PredictVector:
    def __init__(self, embedding_model_name, embedding_model_type, embedding_model_path):
        """
        Inicializa el predictor cargando el modelo de embeddings especificado.

        Parámetros:
            embedding_model_name (str): Nombre del modelo de embeddings.
            embedding_model_type (str): Tipo de modelo ('use', 'sentence_transformer', etc.).
            embedding_model_path (str): Ruta o URL del modelo de embeddings.
        """
        self.manager = EmbeddingModelManager()
        self.manager.load_model(embedding_model_name, embedding_model_type, embedding_model_path)
        self.model_name = embedding_model_name

    def generar_embedding(self, entrada):
        """
        Genera el embedding de una lista de atributos concatenando los elementos y aplicando el modelo de embeddings.

        Parámetros:
            entrada (list): Lista de strings o valores a convertir en embedding.

        Retorna:
            np.ndarray: Vector embedding resultante.
        """
        # texto = entrada.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()
        embedding = self.manager.embed(self.model_name, [entrada])[0]
        return embedding

    def predecir_grupo(self, embedding, umap_path, cluster_model_path):
        """
        Reduce el embedding con UMAP y predice el grupo usando el modelo de clustering (HDBSCAN).

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            umap_path (str): Ruta al modelo UMAP entrenado (pkl).
            cluster_model_path (str): Ruta al modelo de clustering entrenado (pkl).

        Retorna:
            label (int): Etiqueta del grupo predicho (-1 si es outlier).
            embedding_umap (np.ndarray): Embedding reducido por UMAP.
            cluster_model (objeto): Modelo de clustering cargado.
        """
        with open(umap_path, "rb") as f:
            umap_model = pickle.load(f)
        with open(cluster_model_path, "rb") as f:
            cluster_model = pickle.load(f)
        embedding_umap = umap_model.transform([embedding])
        # Para HDBSCAN, usar .predict si está disponible, si no, usar .labels_
        if hasattr(cluster_model, "predict"):
            label = cluster_model.predict(embedding_umap)[0]
        else:
            label = cluster_model.labels_[0]
        return label, embedding_umap
    def existe_en_grupo(self, embedding, embeddings_path, labels_path, grupo_id, atol=1e-6):
        """
        Verifica si el embedding de entrada existe exactamente (o casi exactamente) en el grupo especificado.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_path (str): Ruta al archivo .npy con los embeddings de referencia.
            labels_path (str): Ruta al archivo .npy o .csv con las etiquetas de grupo.
            grupo_id (int): Etiqueta del grupo a buscar.
            atol (float): Tolerancia para igualdad numérica (por defecto 1e-6).

        Retorna:
            existe (bool): True si el embedding está en el grupo, False si no.
            idx (int o None): Índice donde se encontró, o None si no está.
        """
        embeddings = np.load(embeddings_path)
        # Cargar etiquetas (puede ser .npy o .csv)
        if labels_path.endswith('.npy'):
            labels = np.load(labels_path)
        else:
            import pandas as pd
            labels = pd.read_csv(labels_path).values.squeeze()
        # Filtrar embeddings del grupo
        grupo_embeddings = embeddings[labels == grupo_id]
        for idx, emb in enumerate(grupo_embeddings):
            if np.allclose(embedding, emb, atol=atol):
                return True, idx
        return False, None
    
    def existe_en_grupo_por_etiqueta(self, embedding, embeddings_labeled_path, grupo_id, atol=1e-6):
        """
        Verifica si el embedding de entrada existe exactamente (o casi exactamente) en el grupo especificado
        dentro del conjunto completo de embeddings etiquetados.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_labeled_path (str): Ruta al archivo .npy con todos los embeddings etiquetados.
            grupo_id (int): Etiqueta del grupo donde buscar.
            atol (float): Tolerancia para igualdad numérica (por defecto 1e-6).

        Retorna:
            existe (bool): True si el embedding está en el grupo, False si no.
            idx (int o None): Índice relativo dentro del grupo donde se encontró, o None si no está.
            idx_global (int o None): Índice global en el archivo original, o None si no está.
        """
        # Cargar embeddings etiquetados (.npy que incluye la columna 'label')
        embeddings_labeled = np.load(embeddings_labeled_path)
        
        # Separar los embeddings de las etiquetas (asumiendo que la última columna es 'label')
        embeddings = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última
        labels = embeddings_labeled[:, -1]       # Última columna (etiquetas)
        
        # Filtrar solo los embeddings del grupo especificado
        indices_grupo = np.where(labels == grupo_id)[0]
        if len(indices_grupo) == 0:
            return False, None, None
        
        grupo_embeddings = embeddings[indices_grupo]
        
        # Buscar si el embedding existe en el grupo
        for idx_relativo, emb in enumerate(grupo_embeddings):
            if np.allclose(embedding, emb, atol=atol):
                idx_global = indices_grupo[idx_relativo]
                return True, idx_relativo, idx_global
        
        return False, None, None

    def buscar_similares_en_grupo_por_etiqueta(self, embedding, embeddings_labeled_path, grupo_id, top_n=10):
        """
        Busca los top_n embeddings más similares al embedding dado, pero solo dentro del grupo especificado
        del archivo de embeddings etiquetados.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_labeled_path (str): Ruta al archivo .npy con todos los embeddings etiquetados.
            grupo_id (int): Etiqueta del grupo a buscar.
            top_n (int): Número de vecinos más similares a retornar.

        Retorna:
            top_idx (np.ndarray): Índices relativos dentro del grupo de los embeddings más similares.
            similarities (np.ndarray): Valores de similitud correspondientes.
            idx_global (np.ndarray): Índices globales en el archivo de embeddings etiquetados.
        """
        # Cargar embeddings etiquetados (.npy que incluye la columna 'label')
        embeddings_labeled = np.load(embeddings_labeled_path)
        
        # Separar los embeddings de las etiquetas (asumiendo que la última columna es 'label')
        embeddings = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última
        labels = embeddings_labeled[:, -1]       # Última columna (etiquetas)
        
        # Filtrar embeddings y obtener índices globales del grupo
        idx_global = np.where(labels == grupo_id)[0]
        if len(idx_global) == 0:
            return np.array([]), np.array([]), np.array([])
        
        grupo_embeddings = embeddings[idx_global]
        
        # Calcular similitudes coseno
        similarities = cosine_similarity([embedding], grupo_embeddings)[0]
        
        # Obtener los top_n más similares
        top_idx = similarities.argsort()[-top_n:][::-1]
        
        return top_idx, similarities[top_idx], idx_global[top_idx]

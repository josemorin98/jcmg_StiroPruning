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
        texto = entrada.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()
        embedding = self.manager.embed(self.model_name, [texto])[0]
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
    
    
    def buscar_similares_en_grupo(self, embedding, embeddings_path, labels_path, grupo_id, top_n=10):
        """
        Busca los top_n embeddings más similares al embedding dado, pero solo dentro del grupo especificado.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_path (str): Ruta al archivo .npy con los embeddings de referencia.
            labels_path (str): Ruta al archivo .npy o .csv con las etiquetas de grupo.
            grupo_id (int): Etiqueta del grupo a buscar.
            top_n (int): Número de vecinos más similares a retornar.

        Retorna:
            top_idx (np.ndarray): Índices relativos dentro del grupo de los embeddings más similares.
            similarities (np.ndarray): Valores de similitud correspondientes.
            idx_global (np.ndarray): Índices globales en el arreglo original de embeddings.
        """
        embeddings = np.load(embeddings_path)
        # Cargar etiquetas (puede ser .npy o .csv)
        if labels_path.endswith('.npy'):
            labels = np.load(labels_path)
        else:
            import pandas as pd
            labels = pd.read_csv(labels_path).values.squeeze()
        # Filtrar embeddings y obtener índices globales del grupo
        idx_global = np.where(labels == grupo_id)[0]
        grupo_embeddings = embeddings[idx_global]
        if len(grupo_embeddings) == 0:
            return np.array([]), np.array([]), np.array([])
        similarities = cosine_similarity([embedding], grupo_embeddings)[0]
        top_idx = similarities.argsort()[-top_n:][::-1]
        return top_idx, similarities[top_idx], idx_global[top_idx]


# def extraer_datos_originales_por_grupo(grupo_id, cluster_data_path, original_csv_path, output_path):
#     """
#     Extrae las filas originales del CSV original correspondientes a los índices del grupo predicho,
#     las guarda en un nuevo archivo CSV y retorna el DataFrame filtrado.

#     Parámetros:
#         grupo_id (int): Etiqueta del grupo predicho.
#         cluster_data_path (str): Ruta al CSV con los vectores y etiquetas de grupo.
#         original_csv_path (str): Ruta al CSV original con los datos completos.
#         output_path (str): Ruta donde se guardarán los datos originales filtrados.

#     Retorna:
#         datos_filtrados (pd.DataFrame): DataFrame con los datos originales del grupo.
#     """
#     # Cargar los datos de clusters y los datos originales
#     df_clusters = pd.read_csv(cluster_data_path)
#     df_original = pd.read_csv(original_csv_path)

#     # Obtener los índices de las filas que pertenecen al grupo_id
#     indices_grupo = df_clusters[df_clusters["KMeans_Label"] == grupo_id].index

#     # Extraer las filas originales usando los índices
#     datos_filtrados = df_original.loc[indices_grupo]

#     # Guardar el resultado
#     datos_filtrados.to_csv(output_path, index=False)
#     print(f"Datos originales del grupo {grupo_id} guardados en '{output_path}'")

#     return datos_filtrados

# def predecir_grupo_y_extraer(embedding, model_path, scaler_path, data_path):
#     """
#     Escala el embedding, predice el grupo con KMeans y extrae los vectores similares del grupo.

#     Parámetros:
#         embedding (np.ndarray): Vector embedding de entrada.
#         model_path (str): Ruta al modelo KMeans entrenado (pkl).
#         scaler_path (str): Ruta al scaler entrenado (pkl).
#         data_path (str): Ruta al archivo CSV con los vectores y etiquetas de grupo.

#     Retorna:
#         grupo (int): Etiqueta de grupo predicha.
#         resultados (pd.DataFrame): DataFrame con los vectores del grupo predicho.
#     """
#     # Cargar scaler y modelo
#     with open(scaler_path, "rb") as f:
#         scaler = pickle.load(f)
#     with open(model_path, "rb") as f:
#         modelo = pickle.load(f)

#     # Escalar el embedding
#     embedding_scaled = scaler.transform([embedding])

#     # Predecir el grupo
#     grupo = modelo.predict(embedding_scaled)[0]

#     # Cargar los datos y extraer los del grupo predicho
#     df = pd.read_csv(data_path)
#     resultados = df[df["KMeans_Label"] == grupo]

#     return grupo, resultados

# def generar_embedding_desde_lista(entrada, name,modelo=None):
#     """
#     Genera el embedding de una lista de atributos concatenando los elementos y aplicando el modelo de embeddings.

#     Parámetros:
#         entrada (list): Lista de strings o valores a convertir en embedding.
#         modelo (objeto): Modelo de embeddings previamente cargado.

#     Retorna:
#         np.ndarray: Vector embedding resultante.
#     """
#     # Concatenar los elementos de la lista en un solo string, separados por espacio
#     texto = ' '.join([str(x).replace(' ', '_') for x in entrada])
    
#     # Si no se pasa un modelo, aquí deberías cargarlo o lanzar un error
#     if modelo is None:
#         raise ValueError("Debes proporcionar un modelo de embeddings previamente cargado.")
    
#     # Obtener el embedding (ajusta según tu modelo)
#     embedding = modelo.embed(name,[texto])[0]  # Por ejemplo, para USE o SentenceTransformer
#     return embedding



# if __name__ == "__main__":
#     # Paso 1: Vector de entrada
#     entrada = ["Tamaulipas.Tampico", "2000", "Mujeres.45_64", "1.5", "1.3"]

#     # Paso 2: Cargar el modelo de embeddings
#     from model_manager import EmbeddingModelManager
    
#     # Paso 2: Obtener embedding de entrada
#     manager = EmbeddingModelManager()
#     manager.load_model("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4")

#     embedding = generar_embedding_desde_lista(entrada,"use", model=manager)

#     # Paso 3: Predecir grupo y extraer todos los vectores de ese grupo
#     grupo, resultados = predecir_grupo_y_extraer(
#         embedding=embedding,
#         model_path="../test/Modelos/modelo_kmeans.pkl", 
#         scaler_path="../test/Modelos/scaler.pkl",
#         data_path="../test/vectores_clusters.csv"
#     )

#     # Paso 4: Guardar resultados del grupo (vectores similares)
#     resultados_path = f"../test/vectores_grupo_{grupo}.csv"
#     resultados.to_csv(resultados_path, index=False)
#     print(f"\nResultados guardados en '{resultados_path}'")

#     # Paso 5: (Opcional) Extraer datos originales de ese grupo
#     extraer_datos_originales_por_grupo(
#         grupo_id=grupo,
#         cluster_data_path="../test/vectores_clusters.csv",
#         original_csv_path="../data/sample.csv",
#         output_path=f"../test/vectores_grupo_{grupo}_originales.csv"
#     ).head()
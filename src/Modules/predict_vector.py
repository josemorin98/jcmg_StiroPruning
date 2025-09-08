import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
# from Modules.model_manager import EmbeddingModelManager
from src.Modules.model_manager import EmbeddingModelManager
import pandas as pd

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

    def predecir_grupo(self, embedding, embeddings_labeled_path, top_n=1):
        """
        Busca el grupo más probable para el embedding de entrada comparando directamente 
        con los embeddings originales etiquetados.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_labeled_path (str): Ruta al archivo .npy con todos los embeddings etiquetados.
            top_n (int): Número de vecinos más similares a considerar (por defecto 1).

        Retorna:
            label (int): Etiqueta del grupo predicho.
            similarity (float): Similitud coseno con el embedding más similar.
            idx_global (int): Índice global del embedding más similar.
        """
        # Cargar embeddings etiquetados (.npy que incluye la columna 'label')
        embeddings_labeled = np.load(embeddings_labeled_path)
        
        # Separar los embeddings de las etiquetas (asumiendo que la última columna es 'label')
        embeddings = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última
        labels = embeddings_labeled[:, -1]       # Última columna (etiquetas)
        
        # Calcular similitudes coseno con todos los embeddings
        similarities = cosine_similarity([embedding], embeddings)[0]
        
        # Encontrar el embedding más similar
        idx_mas_similar = similarities.argmax()
        similarity_max = similarities[idx_mas_similar]
        label_predicho = int(labels[idx_mas_similar])
        
        return label_predicho, similarity_max, idx_mas_similar

    def predecir_top_grupos(self, embedding, embeddings_labeled_path, top_n=5):
        """
        Busca los top_n grupos más probables para el embedding de entrada.

        Parámetros:
            embedding (np.ndarray): Vector embedding de entrada.
            embeddings_labeled_path (str): Ruta al archivo .npy con todos los embeddings etiquetados.
            top_n (int): Número de grupos top a retornar.

        Retorna:
            top_labels (list): Lista de etiquetas de grupos ordenadas por probabilidad.
            top_similarities (list): Lista de similitudes correspondientes.
            top_indices (list): Lista de índices globales correspondientes.
        """
        # Cargar embeddings etiquetados (.npy que incluye la columna 'label')
        embeddings_labeled = np.load(embeddings_labeled_path)
        
        # Separar los embeddings de las etiquetas (asumiendo que la última columna es 'label')
        embeddings = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última
        labels = embeddings_labeled[:, -1]       # Última columna (etiquetas)
        
        # Calcular similitudes coseno con todos los embeddings
        similarities = cosine_similarity([embedding], embeddings)[0]
        
        # Obtener los top_n más similares
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_similarities = similarities[top_indices]
        top_labels = [int(labels[idx]) for idx in top_indices]
        
        return top_labels, top_similarities.tolist(), top_indices.tolist()

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
    
    def existe_en_grupo_por_etiqueta(self, embedding_query, embeddings_path, grupo_id, atol=1e-6):
        """
        Verifica si el embedding de entrada existe exactamente (o casi exactamente) en el grupo especificado
        dentro del conjunto completo de embeddings etiquetados.

        Parámetros:
            embedding_query (np.ndarray): Vector embedding de entrada.
            embeddings_path (str): Ruta al archivo .npy con todos los embeddings etiquetados.
            grupo_id (int): Etiqueta del grupo donde buscar.
            atol (float): Tolerancia para igualdad numérica (por defecto 1e-6).

        Retorna:
            existe (bool): True si el embedding está en el grupo, False si no.
            idx_relativo (int o None): Índice relativo dentro del grupo donde se encontró, o None si no está.
            idx_global (int o None): Índice global en el archivo original, o None si no está.
        """
        # Cargar embeddings etiquetados
        if embeddings_path.endswith('.npy'):
            embeddings_data = np.load(embeddings_path)
            # Convertir a DataFrame para manejar etiquetas
            embeddings_df = pd.DataFrame(embeddings_data)  
            # Asignar nombres de columnas
            embeddings_df.columns = [f"{i}" for i in range(embeddings_df.shape[1])]
            # Las etiquetas colocarlas con el nombre label
            embeddings_df.columns[-1] = 'label'
            
        elif embeddings_path.endswith('.csv'):
            embeddings_df = pd.read_csv(embeddings_path)
        else:
            raise ValueError("El archivo de embeddings debe ser .npy o .csv")
        
        # Verificar si la columna 'label' existe
        if 'label' not in embeddings_df.columns:
            raise FileNotFoundError(f"No se encontraron etiquetas en {embeddings_path}")
        
        # Filtrar embeddings por grupo y obtener índices globales
        mask_grupo = embeddings_df['label'] == grupo_id
        embeddings_group = embeddings_df[mask_grupo]
        indices_globales = embeddings_df.index[mask_grupo].tolist()
        
        if len(embeddings_group) == 0:
            return False, None, None
        
        # Extraer solo las columnas de embeddings (sin la columna 'label')
        embeddings_group_valores = embeddings_group.drop(columns=['label']).values
        
        # Buscar si el embedding existe en el grupo
        print("Buscando en grupo...")
        for idx_relativo, emb_row in enumerate(embeddings_group_valores):
            if np.allclose(embedding_query, emb_row, atol=atol):
                print("Encontrado")
                idx_global = int(indices_globales[idx_relativo])
                return True, idx_relativo, idx_global

        print("No encontrado")
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
        if embeddings_labeled_path.endswith('.npy'):
            embeddings_data = np.load(embeddings_labeled_path)
            # Convertir a DataFrame para manejar etiquetas
            embeddings_df = pd.DataFrame(embeddings_data)  
            # Asignar nombres de columnas
            embeddings_df.columns = [f"{i}" for i in range(embeddings_df.shape[1])]
            # Las etiquetas colocarlas con el nombre label
            embeddings_df.columns[-1] = 'label'
            
        elif embeddings_labeled_path.endswith('.csv'):
            embeddings_df = pd.read_csv(embeddings_labeled_path)
        else:
            raise ValueError("El archivo de embeddings debe ser .npy o .csv")
        
        # Verificar si la columna 'label' existe
        if 'label' not in embeddings_df.columns:
            raise FileNotFoundError(f"No se encontraron etiquetas en {embeddings_labeled_path}")
        
        # Filtrar embeddings por grupo y obtener índices globales
        mask_grupo = embeddings_df['label'] == grupo_id
        embeddings_group = embeddings_df[mask_grupo]
        indices_globales = embeddings_df.index[mask_grupo].tolist()
        
        if len(embeddings_group) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extraer solo las columnas de embeddings (sin la columna 'label')
        embeddings_group_valores = embeddings_group.drop(columns=['label']).values
        
        print(f"Buscando similares en grupo {grupo_id} con {len(embeddings_group_valores)} embeddings...")
        # Calcular similitudes coseno
        similarities = cosine_similarity([embedding], embeddings_group_valores)[0]
        
        # Obtener los top_n más similares
        top_idx = similarities.argsort()[-top_n:][::-1]
        
        # Convertir indices_globales a numpy array para poder indexar con top_idx
        indices_globales_array = np.array(indices_globales)
        
        return top_idx, similarities[top_idx], indices_globales_array[top_idx], embeddings_group

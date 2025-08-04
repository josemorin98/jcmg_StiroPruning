import os
import warnings
# Configuración para ocultar warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ocultar warnings de TensorFlow
warnings.filterwarnings('ignore')  # Ocultar otros warnings

import numpy as np
import time

class EmbeddingModelManager:
    def __init__(self, save_dir="../test"):
        """
        Inicializa el gestor de modelos de embeddings.

        Parámetros:
            save_dir (str): Ruta relativa o absoluta donde se guardarán los embeddings generados.
        """
        self.save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir))
        os.makedirs(self.save_dir, exist_ok=True)
        self.models = {}

    def load_model(self, name, model_type, model_path=None):
        """
        Carga un modelo de embeddings y lo almacena en el gestor.

        Parámetros:
            name (str): Nombre identificador del modelo.
            model_type (str): Tipo de modelo ('use' o 'sentence_transformer').
            model_path (str): Ruta o identificador del modelo a cargar.
        
        Lanza:
            ValueError: Si el tipo de modelo no es soportado.
        """
        if model_type == "use":
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')  # Solo mostrar errores de TensorFlow
            import tensorflow_hub as hub
            self.models[name] = hub.load(model_path)
        elif model_type == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.models[name] = SentenceTransformer(model_path)
        else:
            raise ValueError("Tipo de modelo no soportado")

    def embed(self, name, sentences, save=True, save_csv=True):
        """
        Genera los embeddings para una lista de sentencias usando el modelo especificado.
        Guarda los embeddings en una subcarpeta con el nombre del modelo si 'save' es True.
        También puede guardar los embeddings en formato CSV si 'save_csv' es True.

        Parámetros:
            name (str): Nombre del modelo a utilizar.
            sentences (list): Lista de textos a procesar.
            save (bool): Si es True, guarda los embeddings en formato .npy.
            save_csv (bool): Si es True, guarda los embeddings en formato .csv.

        Retorna:
            np.ndarray: Array de embeddings generados.
        """
        model = self.models[name]
        if name == "use":
            embeddings = np.array(model(sentences))
        else:
            embeddings = model.encode(sentences)
        if save or save_csv:
            print(f'Creando carpeta de {name}')
            model_dir = os.path.join(self.save_dir, name)
            os.makedirs(model_dir, exist_ok=True)
        if save:
            save_path = os.path.join(model_dir, f"{name}.npy")
            np.save(save_path, embeddings)
        if save_csv:
            save_csv_path = os.path.join(model_dir, f"{name}.csv")
            # Si los embeddings son 2D, guarda como DataFrame
            import pandas as pd
            df = pd.DataFrame(embeddings)
            df.to_csv(save_csv_path, index=False)
        return embeddings
import argparse
import numpy as np
from Modules.predict_vector import PredictVector
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Modules.model_manager import EmbeddingModelManager
import time 

def main():
    parser = argparse.ArgumentParser(description="Predicción de grupo para un vector de entrada.")
    # parser.add_argument("--vector", type=str, required=True, help="Vector de entrada separado por comas (ej: 0.1,0.2,0.3,...)")
    parser.add_argument("--modelo", type=str, choices=["random", "bayesiano"], required=True, help="Modelo de clustering a usar")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Ruta al archivo .npy de embeddings")
    parser.add_argument("--modelo_dir", type=str, default="../test/Modelos", help="Directorio donde están los modelos")
    args = parser.parse_args()

    # 1. Inicializa el predictor con el modelo de embeddings que usaste para generar los embeddings
    modelos_dict = {
        "use": ("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4"),
        "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
    }
    nombre, tipo, ruta = modelos_dict[args.modelo]
    predictor = PredictVector(
        embedding_model_name=nombre,  # o el nombre que corresponda
        embedding_model_type=tipo,  # o "sentence_transformer", etc.
        embedding_model_path=ruta  # o la ruta local
    )

    # 2. Prepara tu entrada como lista de atributos (ejemplo)
    entrada = ["Tamaulipas.Tampico", "2000", "Mujeres.45_64", "1.5", "1.3"]

    # 3. Genera el embedding
    embedding = predictor.generar_embedding(entrada)

    # 4. Define las rutas a los modelos UMAP y HDBSCAN (elige random o bayesiano)
    umap_path = f"../test/Modelos_{args.modelo}/umap_bayesian.pkl"
    hdbscan_path = f"../test/Modelos_{args.modelo}/hdbscan_bayesian.pkl"

    # 5. Predice el grupo
    label, embedding_umap = predictor.predecir_grupo(
        embedding, umap_path, hdbscan_path
    )
    print(f"Grupo predicho: {label}")

    # 6. Si es outlier (grupo -1), busca los 10 más similares
    if label == -1:
        embeddings_path = "../test/embeddings/use/use.npy"  # Ajusta la ruta
        top_idx, similarities = predictor.buscar_similares(embedding, embeddings_path, top_n=10)
        print("Índices de los 10 más similares:", top_idx)
        print("Similitudes:", similarities)
    
    
    
    
    
    
    
    # 1. Recibe el vector y lo convierte en oración
    print("Recibiendo vector de entrada...")
    vector_input = ["Jalisco.Zapopan", "2021", "Hombres.Total", "0.2810551936189229","0.2810551936189229"]
    print("Vector de entrada:", vector_input,"\nCambiando a oración...")
    query = vector_input.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()

    

    # 2. Obtiene el embedding usando EmbeddingModelManager
    print("Generar embedding...")
    predictor = EmbeddingModelManager()
    path_test  = "../test/query_embeddings"
    manager = EmbeddingModelManager(save_dir=f"../{path_test}/embeddings")

    # Selección y carga del modelo
    modelos_dict = {
        "use": ("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4"),
        "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
    }
    nombre, tipo, ruta = modelos_dict[args.modelo]
    print(f"\nCargando modelo {nombre}...")

    start_time = time.time()
    manager.load_model(nombre, tipo, ruta)
    tiempo_carga_modelo = time.time() - start_time
    print(f"Modelo cargado en {tiempo_carga_modelo:.2f} segundos.")

    # Generar y guardar embeddings
    start_time = time.time()
    print(f"\nGenerando y guardando embedding para {nombre} del vector de entrada...")
    embedding_vector = manager.embed(nombre, query, save=True, save_csv=True)
    tiempo_embeddings = time.time() - start_time
    print(f"Embeddings generados y guardados en {tiempo_embeddings:.2f} segundos.")
    embedding = embedding_vector[0]  # Asumiendo que es un array de un solo embedding
    print("Embedding generado:", embedding)

    # 3. Carga el modelo de predicción
    print("Cargando modelo de predicción...")
    predictor = PredictVector(
        # scaler_path=f"{args.modelo_dir}/hdbscan_bayesian.pkl",
        model_path=f"{args.modelo_dir}/.pkl",
        data_path=f"{args.modelo_dir}/data.csv"
    )





    # 3. Carga el mejor modelo guardado
    if args.modelo == "random":
        modelo_path = f"{args.modelo_dir}/hdbscan_random.pkl"
        umap_path = f"{args.modelo_dir}/umap_random.pkl"
    else:
        modelo_path = f"{args.modelo_dir}/hdbscan_bayesian.pkl"
        umap_path = f"{args.modelo_dir}/umap_bayesian.pkl"

    with open(umap_path, "rb") as f:
        umap_model = pickle.load(f)
    with open(modelo_path, "rb") as f:
        hdbscan_model = pickle.load(f)

    # 4. Reduce el embedding con UMAP
    embedding_umap = umap_model.transform([embedding])

    # 5. Predice el grupo
    label = hdbscan_model.predict(embedding_umap)[0]

    print(f"Grupo asignado: {label}")

    # 6. Si no pertenece a ningún grupo, buscar los 10 más similares
    if label == -1:
        print("No se encontró grupo. Buscando los 10 más similares...")
        embeddings = np.load(args.embeddings_path)
        similarities = cosine_similarity([embedding], embeddings)[0]
        top10_idx = similarities.argsort()[-10:][::-1]
        print("Índices de los 10 más similares:", top10_idx)
        print("Similitudes:", similarities[top10_idx])
        print("Índice con mayor similitud:", top10_idx[0], "Similitud:", similarities[top10_idx[0]])

if __name__ == "__main__":
    main()
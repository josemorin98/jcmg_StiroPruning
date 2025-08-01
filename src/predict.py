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
    parser.add_argument("--modelo", type=str, choices=["use", "st1", "st2", "st3"], help="Modelo de clustering a usar", default="st1")
    parser.add_argument("--params", type=str, choices=["random", "bayesiano"], help="Modelo de clustering a usar", default="bayesiano")
    parser.add_argument("--embeddings_path", type=str, default="../test/embeddings", help="Ruta al archivo .npy de embeddings")
    parser.add_argument("--params_dir", type=str, default="../test/Modelos", help="Directorio donde están los modelos")
    args = parser.parse_args()
    
    # 1. Recibe el vector y lo convierte en oración
    print("Recibiendo vector de entrada...")
    vector_input = ["Jalisco.Zapopan", "2021", "Hombres.Total", "0.2810551936189229","0.2810551936189229"]
    print("Vector entrada:", vector_input)
    # query = vector_input.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()
    query_string = ' '.join([str(x).replace(' ', '_') for x in vector_input])
    print("Vector en sentencia:", query_string)

   # 2. Selección del modelo de embeddings
    modelos_dict = {
        "use": ("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4"),
        "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
    }
    nombre, tipo, ruta = modelos_dict[args.modelo]
    print(f"Cargando modelo de embeddings: {nombre}")

    predictor = PredictVector(nombre, tipo, ruta)

    # 3. Generar embedding
    embedding = predictor.generar_embedding(query_string)
    print("Embedding generado.")

    # exit()
    # 4. Selección de modelos UMAP y HDBSCAN según el tipo de parámetros
    if args.params == "random":
        umap_path = f"{args.params_dir}_{nombre}/umap_random.pkl"
        hdbscan_path = f"{args.params_dir}_{nombre}/hdbscan_random.pkl"
    else:
        umap_path = f"{args.params_dir}_{nombre}/umap_bayesian.pkl"
        hdbscan_path = f"{args.params_dir}_{nombre}/hdbscan_bayesian.pkl"

    # 5. Predicción de grupo
    label, embedding_umap = predictor.predecir_grupo(embedding, umap_path, hdbscan_path)
    print(f"Grupo asignado: {label}")
    # exit()
    
    # 6. Verificar si el embedding ya existe en el grupo
    embeddings_labeled_path = f"{args.params_dir}_{nombre}/embeddings_labeled.npy"

    existe, idx_relativo, idx_global = predictor.existe_en_grupo_por_etiqueta(
    embedding_umap, embeddings_labeled_path, label
)
    if existe:
        print(f"El embedding ya existe en el grupo {label}, índice relativo: {idx_relativo}, índice global: {idx_global}")
    else:
        print(f"El embedding NO existe en el grupo {label}")

        # 7. Si es outlier, buscar los 10 más similares en todos los embeddings
        if label == -1:
            print("No se encontró grupo. Buscando los 10 más similares en todos los embeddings...")
            from sklearn.metrics.pairwise import cosine_similarity
            embeddings = np.load(embeddings_labeled_path)
            similarities = cosine_similarity([embedding], embeddings)[0]
            top10_idx = similarities.argsort()[-10:][::-1]
            print("Índices de los 10 más similares:", top10_idx)
            print("Similitudes:", similarities[top10_idx])
            print("Índice con mayor similitud:", top10_idx[0], "Similitud:", similarities[top10_idx[0]])
        else:
            top_idx, similarities, idx_global = predictor.buscar_similares_en_grupo_por_etiqueta(
                embedding, embeddings_labeled_path, label, top_n=10
            )
            
            if len(top_idx) > 0:
                print("Top 10 similares dentro del grupo:")
                print("Índices relativos en el grupo:", top_idx)
                print("Índices globales en el archivo:", idx_global)
                print("Similitudes:", similarities)
            else:
                print(f"No se encontraron embeddings en el grupo {label}")
    

if __name__ == "__main__":
    main()
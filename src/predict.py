import argparse
import numpy as np
import pandas as pd
from Modules.predict_vector import PredictVector
from Modules.classification_manager import ClassificationManager
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Modules.model_manager import EmbeddingModelManager
import time 

def main():
    parser = argparse.ArgumentParser(description="Predicción de grupo usando clasificación para un vector de entrada.")
    parser.add_argument("--modelo", type=str, choices=["use", "st1", "st2", "st3"], help="Modelo de embeddings a usar", default="st1")
    parser.add_argument("--params", type=str, choices=["random", "bayesian", "spearate_grid"], help="Tipo de parámetros de clustering usados", default="bayesian")
    parser.add_argument("--embeddings_path", type=str, default="../test/embeddings", help="Ruta al archivo .npy de embeddings")
    parser.add_argument("--models_dir", type=str, default="../test/Modelos", help="Directorio donde están los modelos")
    parser.add_argument("--use_adjusted", action="store_true", help="Usar embeddings ajustados con columnas spatial, temporal e interest")
    args = parser.parse_args()
    
    # Determinar sufijo según si se usan embeddings ajustados
    model_suffix = "_adjusted" if args.use_adjusted else ""
    params_suffix = f"_{args.params}"
    
    print(f"Usando modelo: {args.modelo}")
    print(f"Tipo de parámetros: {args.params}")
    print(f"Embeddings ajustados: {args.use_adjusted}")
    
    # 1. Vector de entrada (mismo formato que antes)
    print("\n=== VECTOR DE ENTRADA ===")
    vector_input = ["Mexico.Total", "2012", "Mujeres.>65", "0.139", "0.139"]
    print("Vector entrada:", vector_input)
    if args.use_adjusted:
        # Para embeddings ajustados, agregar las columnas spatial, temporal e interest
        vector_input.extend(["spatial", "temporal", "interest"])
        print("Vector entrada ajustado:", vector_input)
    # vector_input = ["Mexico.Total", "2012", "Mujeres.>65", "0.2810551936189229", "0.2810551936189229"]
    query_string = ' '.join([str(x).replace(' ', '_') for x in vector_input])
    print("Vector en sentencia:", query_string)

    # 2. Selección del modelo de embeddings
    print("\n=== GENERACIÓN DE EMBEDDING ===")
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
    print("Embedding generado con dimensiones:", embedding.shape)

    # 4. Inicializar el gestor de clasificación
    print("\n=== CLASIFICACIÓN ===")
    classifier_manager = ClassificationManager(random_state=42)
    
    # 5. Cargar datos de clustering para entrenar el clasificador
    models_dir = f"{args.models_dir}_{args.modelo}{model_suffix}"
    # embeddings_path = f"../test/Modelos_{args.modelo}{model_suffix}/embeddings_originales_labeled.npy"
    
    # Cargar embeddings etiquetados para búsqueda
    # embeddings_umap_labeled_path = f"{models_dir}/embeddings_umap_labeled_{args.params}.npy"
    embeddings_originales_labeled_path = f"{models_dir}/embeddings_originales_labeled_{args.params}.npy"
    print(f"Cargando datos desde: {models_dir}")
    print(f"Cargando embeddings origianles etiquetados desde: {embeddings_originales_labeled_path}")
    
    embeddings_data, labels = classifier_manager.load_clustering_data(
        embeddings_path=embeddings_originales_labeled_path,
        clustering_manager_dir=models_dir,
        method=args.params
    )
    
    print(f"Datos cargados: {len(embeddings_data)} muestras, {len(np.unique(labels))} clases únicas")
    
    # 6. Preparar datos para clasificación
    X_train, X_test, y_train, y_test = classifier_manager.prepare_classification_data(
        embeddings_data, labels, remove_noise=True
    )
    
    # 7. Entrenar clasificadores
    print("Entrenando clasificadores...")
    results = classifier_manager.train_classifiers(X_train, y_train, cv_folds=3)
    
    # 8. Predecir grupo para el nuevo vector
    best_model = classifier_manager.best_model
    predictions, probabilities = classifier_manager.predict_new_data(best_model, embedding.reshape(1, -1))
    predicted_label = predictions[0]
    
    print(f"\n=== RESULTADO DE PREDICCIÓN ===")
    print(f"Mejor modelo: {best_model}")
    print(f"Grupo predicho: {predicted_label}")
    
    if probabilities is not None:
        max_prob = np.max(probabilities[0])
        print(f"Probabilidad máxima: {max_prob:.4f}")
        
        # Mostrar probabilidades de todas las clases
        unique_labels = np.unique(y_train)
        print("Probabilidades por clase:")
        for i, label in enumerate(unique_labels):
            print(f"  Clase {label}: {probabilities[0][i]:.4f}")

    # 9. Reutilizar funciones de búsqueda exacta o top 10 más similar
    print(f"\n=== BÚSQUEDA EN GRUPO {predicted_label} ===")
    
    # Verificar si el embedding ya existe exactamente en el grupo
    existe, idx_relativo, idx_global = predictor.existe_en_grupo_por_etiqueta(
        embedding, embeddings_originales_labeled_path, predicted_label
    )
    
    if existe:
        print(f"El embedding ya existe en el grupo {predicted_label}")
        print(f"Índice relativo en el grupo: {idx_relativo}")
        print(f"Índice global en el dataset: {idx_global}")
    else:
        print(f"El embedding NO existe en el grupo {predicted_label}")
        
        # Si es outlier/ruido, buscar los 10 más similares en todos los embeddings
        if predicted_label == -1:
            print("Grupo predicho es ruido (-1). Buscando los 10 más similares en todos los embeddings...")
            # Cargar embeddings originales etiquetados
            embeddings_labeled = np.load(embeddings_originales_labeled_path)
            embeddings_all = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última (etiquetas)
            similarities = cosine_similarity([embedding], embeddings_all)[0]
            top10_idx = similarities.argsort()[-10:][::-1]
            
            print("Top 10 más similares:")
            for i, idx in enumerate(top10_idx):
                # Obtener la etiqueta del embedding similar
                label_similar = int(embeddings_labeled[idx, -1])
                print(f"  {i+1}. Índice: {idx}, Similitud: {similarities[idx]:.4f}, Grupo: {label_similar}")
                
        else:
            # Buscar similares dentro del grupo predicho
            top_idx, similarities, idx_global = predictor.buscar_similares_en_grupo_por_etiqueta(
                embedding, embeddings_originales_labeled_path, predicted_label, top_n=10
            )
            
            if len(top_idx) > 0:
                print(f"Top {len(top_idx)} similares dentro del grupo {predicted_label}:")
                for i, (rel_idx, glob_idx, sim) in enumerate(zip(top_idx, idx_global, similarities)):
                    print(f"  {i+1}. Índice relativo: {rel_idx}, Índice global: {glob_idx}, Similitud: {sim:.4f}")
            else:
                print(f"No se encontraron embeddings en el grupo {predicted_label}")
    
    print(f"\n=== RESUMEN ===")
    print(f"Vector de entrada procesado exitosamente")
    print(f"Modelo de embeddings: {nombre}")
    print(f"Modelo de clasificación: {best_model}")
    print(f"Grupo predicho: {predicted_label}")
    if probabilities is not None:
        print(f"Confianza: {max_prob:.4f}")

if __name__ == "__main__":
    main()
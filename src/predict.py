import argparse
import numpy as np
import pandas as pd
from Modules.predict_vector import PredictVector
from Modules.classification_manager import ClassificationManager

def main():
    parser = argparse.ArgumentParser(description="Predicción de grupo usando clasificación para un vector de entrada.")
    parser.add_argument("--modelo", type=str, choices=["use", "st1", "st2", "st3"], help="Modelo de embeddings a usar", default="st1")
    parser.add_argument("--params", type=str, choices=["random", "bayesian", "separate_grid"], help="Tipo de parámetros de clustering usados", default="bayesian")
    parser.add_argument("--embeddings_path", type=str, default="test/Embeddings", help="Ruta al archivo .npy de embeddings")
    parser.add_argument("--models_dir", type=str, default="test/Modelos", help="Directorio donde están los modelos")
    parser.add_argument("--use_adjusted", action="store_true", help="Usar embeddings ajustados con columnas spatial, temporal e interest")
    args = parser.parse_args()
    
    # Determinar sufijo según si se usan embeddings ajustados
    model_suffix = "_adjusted" if args.use_adjusted else ""
    params_suffix = f"_{args.params}"
    if args.params == "separate_grid":
        model_params = "gridSearch"
    
    print(f"Usando modelo: {args.modelo}")
    print(f"Tipo de parámetros: {args.params}")
    print(f"Embeddings ajustados: {args.use_adjusted}")
    
    # 1. Vector de entrada (mismo formato que antes)
    print("\n=== VECTOR DE ENTRADA ===")
    # vector_input = ["Mexico.Total", "2012", "Mujeres.>65", "0.139", "0.139"]   # PRRUEBA 1
    vector_input = ["zacatecas.benito juarez", "2017", "c80.mujeres.total",	"1k", "0.4739336492890995"]  # PRUEBA 2 Exacta
    vector_input = ["tabasco.total", "2023", "c64.hombres.total", "100k", "3.594174562"]  # PRUEBA 2.1 No Exacta

    vector_input = ["aguascalientes", "2004", "arsenico_(polvos_respirables_vapores_o_humos)_aire",	989.0, 1]  # PRUEBA 3 Exacta
    vector_input = ["aguascalientes", "2004", "arsenico_(polvos_respirables_vapores_o_humos)_agua",	989.0, 1]  # PRUEBA 3.1 No Exacta

    print("Vector entrada:", vector_input)
    if args.use_adjusted:
        # Para embeddings ajustados, usar formato con spatial, temporal e interest
        spatial      =  vector_input[0]  # Ejemplo de valor para spatial
        temporal     =  vector_input[1]  # Ejemplo de valor para temporal
        interest     =  vector_input[2]  # Ejemplo de valor para interest
        reference      =  vector_input[3]  # Ejemplo de valor para referencia
        observable   =  vector_input[4]  # Ejemplo de valor para observable
        vector_input =  [spatial, temporal, interest] # Mantener los valores numéricos
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
    embeddings_originales_labeled_path = f"{models_dir}/{model_params}/HDBSCAN/embeddings_labeled_{args.params}.csv"
    print(f"Cargando datos desde: {models_dir}")
    print(f"Cargando embeddings origianles etiquetados desde: {embeddings_originales_labeled_path}")
    
    embeddings_data, labels = classifier_manager.load_data(
        embeddings_path=embeddings_originales_labeled_path
    )
    
    print(f"Datos cargados: {len(embeddings_data)} muestras, {len(np.unique(labels))} clases únicas")
    
    
    # 6. Preparar datos para clasificación
    # X_train, X_test, y_train, y_test = classifier_manager.prepare_classification_data(
    #     embeddings_data, labels, remove_noise=False
    # )
    
    # print(f"Datos de entrenamiento: {X_train.shape}, Etiquetas: {y_train.shape}")
    
    # 7. Entrenar clasificadores
    print("----- Entrenando clasificadores -----")
    results = classifier_manager.train_classifiers(embeddings_data, labels, cv_folds=3)
    
    # 8. Predecir grupo para el nuevo vector
    best_model = classifier_manager.best_model
    predictions, probabilities = classifier_manager.predict_new_data(best_model, embedding.reshape(1, -1))
    predicted_label = predictions[0]
    
    # exit()
    print(f"\n=== RESULTADO DE PREDICCIÓN ===")
    print(f"Mejor modelo: {best_model}")
    print(f"Grupo predicho: {predicted_label}")
    
    if probabilities is not None:
        max_prob = np.max(probabilities[0])
        print(f"Probabilidad máxima: {max_prob:.4f}")
        
        # Mostrar probabilidades de todas las clases
        print("Probabilidades por clase:")
        
        unique_labels = np.unique(labels)
        probabilities = probabilities[0].tolist()
        
        df_probabilities = pd.DataFrame({
            'label': unique_labels,
            'probability': probabilities
        }).sort_values(by='probability', ascending=False)
        
        print(df_probabilities.to_string(index=False))
        
    # 9. Reutilizar funciones de búsqueda exacta o top 10 más similar
    print(f"\n=== BÚSQUEDA EN GRUPO {predicted_label} ===")
    
    # Verificar si el embedding ya existe exactamente en el grupo
    existe, idx_relativo, idx_global, grupo_existente = predictor.existe_en_grupo_por_etiqueta(
        embedding, embeddings_originales_labeled_path, predicted_label
    )
    
    if existe:
        print(f"El embedding ya existe en el grupo {predicted_label}")
        print(f"Índice relativo en el grupo: {idx_relativo}")
        print(f"Índice global en el dataset: {idx_global}")
        
        # Cargar el CSV original para mostrar el vector correspondiente
        try:
            csv_original = pd.read_csv("../data/sample.csv")
            if idx_global < len(csv_original):
                vector_original = csv_original.iloc[idx_global]
                print(f"\nVector original encontrado:")
                print(f"   Spatial: {vector_original['spatial']}")
                print(f"   Temporal: {vector_original['temporal']}")
                print(f"   Interest: {vector_original['interest']}")
                print(f"   Reference: {vector_original['reference']}")
                print(f"   Observation: {vector_original['observation']}")
                
                # Mostrar la sentencia como se procesaría
                if args.use_adjusted:
                    sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']}".replace(' ', '_')
                else:
                    sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']} {vector_original['reference']} {vector_original['observation']}".replace(' ', '_')
                print(f"Sentencia procesada: {sentencia_procesada}")
            else:
                print(f"Índice global {idx_global} fuera del rango del CSV original")
        except Exception as e:
            print(f"Error al cargar CSV original: {str(e)}")
    else:
        print(f"El embedding NO existe en el grupo {predicted_label}")
        
        # Si es outlier/ruido, buscar los 10 más similares en todos los embeddings
        if predicted_label == False:
            print("Grupo predicho es ruido (False). Buscando los 10 más similares en todos los embeddings...")
            # Cargar embeddings originales etiquetados
            
            # Buscar los 10 más similares en todos los embeddings o buscar entre el rudio
            
            
            # if embeddings_originales_labeled_path.endswith('.npy'):
            #     embeddings_labeled = np.load(embeddings_originales_labeled_path)
            #     # Convertir a DataFrame para manejar etiquetas
            #     embeddings_labeled = pd.DataFrame(embeddings_labeled)  
            #     # Asignar nombres de columnas
            #     embeddings_labeled.columns = [f"{i}" for i in range(embeddings_labeled.shape[1])]
            #     # Las etiquetas colocarlas con el nombre label
            #     embeddings_labeled.columns[-1] = 'label'
            # elif embeddings_originales_labeled_path.endswith('.csv'):
            #     embeddings_labeled = pd.read_csv(embeddings_originales_labeled_path)  # Convertir a numpy array
            # else:
            #     raise ValueError("El archivo de embeddings debe ser .npy o .csv")
        
            # embeddings_all = embeddings_labeled[:, :-1]  # Todas las columnas excepto la última (etiquetas)
            # similarities = cosine_similarity([embedding], embeddings_all)[0]
            # top10_idx = similarities.argsort()[-10:][::-1]
            
            # print("Top 10 más similares:")
            # for i, idx in enumerate(top10_idx):
            #     # Obtener la etiqueta del embedding similar
            #     label_similar = int(embeddings_labeled[idx, -1])
            #     print(f"  {i+1}. Índice: {idx}, Similitud: {similarities[idx]:.4f}, Grupo: {label_similar}")
                
        else:
            # Buscar similares dentro del grupo predicho
            top_idx, similarities, idx_global, embeddings_group = predictor.buscar_similares_en_grupo_por_etiqueta(
                embedding, embeddings_originales_labeled_path, predicted_label, top_n=10
            )

            embeddings_group.to_csv("../data/embeddings_group.csv", index=False)

            if len(top_idx) > 0:
                print(f"Top {len(top_idx)} similares dentro del grupo {predicted_label}:")
                
                # Cargar el CSV original para extraer los vectores originales
                try:
                    csv_original = pd.read_csv("../data/sample_v2.csv")
                    print(f"\nVectores originales correspondientes:")
                    print("="*80)
                    
                    for i, (rel_idx, glob_idx, sim) in enumerate(zip(top_idx, idx_global, similarities)):
                        print(f"\n{i+1}. Similitud: {sim:.4f}")
                        print(f"   Índice relativo: {rel_idx}, Índice global: {glob_idx}")
                        
                        # Verificar que el índice global esté dentro del rango del CSV
                        if glob_idx < len(csv_original):
                            vector_original = csv_original.iloc[glob_idx]
                            print(f"   Vector original:")
                            print(f"     Spatial: {vector_original['spatial']}")
                            print(f"     Temporal: {vector_original['temporal']}")
                            print(f"     Interest: {vector_original['interest']}")
                            print(f"     Reference: {vector_original['reference']}")
                            print(f"     Observation: {vector_original['observation']}")
                            
                            # Mostrar la sentencia como se procesaría
                            if args.use_adjusted:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']}"
                            else:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']} {vector_original['reference']} {vector_original['observation']}"
                            print(f"Sentencia procesada: {sentencia_procesada}")
                        else:
                            print(f"Índice global {glob_idx} fuera del rango del CSV original")
                        print("-" * 60)
                        
                except FileNotFoundError:
                    print("No se pudo cargar el CSV original '../data/sample.csv'")
                except Exception as e:
                    print(f"Error al cargar CSV original: {str(e)}")
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
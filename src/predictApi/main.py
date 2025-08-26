from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from src.Modules.predict_vector import PredictVector
from src.Modules.classification_manager import ClassificationManager
import unicodedata
import os
import json
import datetime

app = FastAPI()
classifier_manager = ClassificationManager(random_state=42)
dataset_originales_path = "../data/sample.csv"

# Modelo de entrada para la API
class PredictRequest(BaseModel):
    modelo: str = "st1"
    # params: str = "bayesian"
    # embeddings_path: str = "test/Embeddings"
    # models_dir: str = "test/Modelos"
    use_adjusted: bool = False
    vector_input: list
    
class TrainRequest(BaseModel):
    modelo: str = "st1"
    params: str = "bayesian"
    embeddings_path: str = "test/Embeddings"
    models_dir: str = "test/Modelos"
    ds_originales_path: str = "./data/sample.csv"
    use_adjusted: bool = False

@app.post("/trainClassifier")
def train_classifier(request: TrainRequest):
    try:
        # Ejemplo de JSON para el request:
        # {
        #     "modelo": "st1",
        #     "params": "bayesian",
        #     "embeddings_path": "test/Embeddings",
        #     "models_dir": "test/Modelos",
        #     "ds_originales_path": "../data/sample.csv",
        #     "use_adjusted": false
        # }
        dataset_originales_path = request.ds_originales_path
        # Aquí iría la lógica para entrenar el clasificador
        # Cargar datos de clustering para entrenar el clasificador
        print("Iniciando entrenamiento del clasificador...")
        print(f"Parámetros recibidos: {request}")

        # Aquí iría la lógica para entrenar el clasificador
        print("Cargando datos de clustering para entrenar el clasificador...")

        model_suffix = "_adjusted" if request.use_adjusted == True else  ""
        params_suffix = f"_{request.params}"
        model_params = request.params
        if request.params == "separate_grid":
            model_params = "gridSearch"
        
        # Crear directorio de modelos si no existe
        os.makedirs(f"{request.models_dir}_{request.modelo}{model_suffix}/{model_params}/HDBSCAN", exist_ok=True)
        models_dir = f"{request.models_dir}_{request.modelo}{model_suffix}"
        embeddings_originales_labeled_path = os.path.abspath(f"{models_dir}/{model_params}/HDBSCAN/embeddings_labeled_{request.params}.csv")

        
        # print(f"Ruta de embeddings etiquetados: {embeddings_originales_labeled_path}")
        embeddings_data, labels = classifier_manager.load_data(
            embeddings_path=embeddings_originales_labeled_path
        )

        # Entrenar clasificadores
        results = classifier_manager.train_classifiers(embeddings_data, labels, cv_folds=3)
        # Guardar resultados en formato requerido
        results_list = []
        for model_name, metrics in results.items():
            results_list.append({
                "embeding_model": request.modelo,
                "classifier_model": model_name,
                "r1": metrics["r1"],
                "accuracy": metrics["accuracy"],
                "cv_mean": metrics["cv_mean"],
                "cv_std": metrics["cv_std"],
                "cv_scores": metrics["cv_scores"]
            })
        results_df = pd.DataFrame(results_list)
        # results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(models_dir, f"train_results.csv")
        if os.path.exists(results_csv_path):
            existing_df = pd.read_csv(results_csv_path)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
        results_df.to_csv(results_csv_path, index=False)

        # Predecir grupo para el nuevo vector

        return {"message": f"Clasificador entrenado exitosamente. Mejor modelo: {classifier_manager.best_model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def limpiar_texto(texto):
    if isinstance(texto, list):
        texto = ' '.join([str(x) for x in texto])
    # Convierte a minúsculas
    texto = texto.lower()
    # Elimina acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    # Reemplaza espacios por guiones bajos
    texto = texto.replace(' ', '_')
    # Elimina comas
    texto = texto.replace(',', '')
    return texto




@app.post("/predict")
def predict(request: PredictRequest):
    # Ejemplo de JSON de entrada para /predict:
    # {
    #     "modelo": "st1",                # Modelo de embeddings a usar: "st1", "st2", "st3" o "use"
    #     "use_adjusted": false,          # Si se usan embeddings ajustados (true/false)
    #     "vector_input": [               # Vector de entrada, puede ser lista de strings o valores
    #         "Madrid",                   # spatial
    #         "2023",                     # temporal
    #         "Temperatura",              # interest
    #         "Referencia",               # reference (opcional, si use_adjusted es false)
    #         "Observación"               # observation (opcional, si use_adjusted es false)
    #     ]
    # }
    try:
        
        # Preparar vector de entrada
        vector_input = request.vector_input
        if request.use_adjusted:
            spatial = vector_input[0]
            temporal = vector_input[1]
            interest = vector_input[2]
            reference = vector_input[3]
            observable = vector_input[4]
            vector_input = [spatial, temporal, interest]
        
        query_string = limpiar_texto(vector_input)
        print(f"Vector de entrada procesado: {query_string}")
        # Selección del modelo de embeddings
        modelos_dict = {
            "use": ("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4"),
            "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
            "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
            "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
        }
        nombre, tipo, ruta = modelos_dict[request.modelo]
        print(f"Modelo seleccionado: {nombre}, Tipo: {tipo}, Ruta: {ruta}")

        predictor = PredictVector(nombre, tipo, ruta)

        # Generar embedding
        embedding = predictor.generar_embedding(query_string)
        print(f"Embedding generado tamaño de: {len(embedding)}")
        # best_model = classifier_manager.best_model
        print(f"Prediccion {classifier_manager.best_model}")
        predictions, probabilities = classifier_manager.predict_new_data(
            model_name=classifier_manager.best_model,
            new_embeddings=embedding.reshape(1, -1)
        )
        predicted_label = predictions[0]

        response = {
            "grupo_predicho": int(predicted_label) if predicted_label is not False else False,
            "modelo": nombre,
            "clasificador": str(classifier_manager.best_model),
        }
        print("Respuesta generada:\n" + json.dumps(response, indent=4, ensure_ascii=False))

        labels = classifier_manager.labels
        embeddings = classifier_manager.embeddings
        embeddings_originales_labeled_path = classifier_manager.embeddings_path
        if probabilities is not None:
            max_prob = float(np.max(probabilities[0]))
            response["confianza"] = max_prob
            unique_labels = np.unique(labels)
            probabilities_list = probabilities[0].tolist()
            # response["probabilidades"] = [
            #     {"label": int(lbl), "probabilidad": float(prob)}
            #     for lbl, prob in zip(unique_labels, probabilities_list)
            # ]

        print(f"Grupo predicho: {predicted_label} con confianza {response.get('confianza', 'N/A')}")
        
        # Verificar si el embedding ya existe exactamente en el grupo
        existe, idx_relativo, idx_global, grupo_existente = predictor.existe_en_grupo_por_etiqueta(
            embedding, embeddings_originales_labeled_path, predicted_label
        )
        response["existe_en_grupo"] = existe
        if existe:
            response["indice_relativo"] = int(idx_relativo)
            response["indice_global"] = int(idx_global)
            # Opcional: cargar vector original si se requiere
            try:
                df_embeddings = pd.read_csv(dataset_originales_path)
                if existe and idx_global is not None:
                    vector_original = df_embeddings.iloc[int(idx_global)].to_dict()
                    response["vector_original"] = vector_original

            except Exception as e:
                response["vector_original_error"] = str(e)
        else:
            # Buscar similares dentro del grupo predicho
            top_idx, similarities, idx_global, embeddings_group = predictor.buscar_similares_en_grupo_por_etiqueta(
                embedding, embeddings_originales_labeled_path, predicted_label, top_n=10)

            if len(top_idx) > 0:
                # Cargar el CSV original para extraer los vectores originales
                print(f"Tamaño de similitudes: {len(similarities)}, indices: {len(top_idx)}")
                try:
                    csv_original = pd.read_csv(dataset_originales_path)
                    top_10 = []
                    for i, (rel_idx, glob_idx, sim) in enumerate(zip(top_idx, idx_global, similarities)):
                        if glob_idx < len(csv_original):
                            vector_original = csv_original.iloc[glob_idx]
                            if request.use_adjusted:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']}"
                            else:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']} {vector_original['reference']} {vector_original['observation']}"
                            top_10.append({
                                "similitud": float(sim),
                                "indice_relativo": int(rel_idx),
                                "indice_global": int(glob_idx),
                                "vector_original": {
                                    "spatial": vector_original['spatial'],
                                    "temporal": vector_original['temporal'],
                                    "interest": vector_original['interest'],
                                    "reference": vector_original['reference'],
                                    "observation": vector_original['observation']
                                },
                                "sentencia_procesada": sentencia_procesada
                            })
                        else:
                            top_10.append({
                                "similitud": float(sim),
                                "indice_relativo": int(rel_idx),
                                "indice_global": int(glob_idx),
                                "error": f"Índice global {glob_idx} fuera del rango del CSV original"
                            })
                    response["top_10"] = top_10
                    # Guardar similitudes en un CSV por consulta

                    # Definir nombre base del archivo
                    csv_base = "similitudes_consulta"
                    csv_dir = "./"
                    consulta_num = 1

                    # Buscar un nombre de archivo que no exista aún
                    while os.path.exists(os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")):
                        consulta_num += 1

                    csv_path = os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")

                    # Preparar los datos para guardar
                    rows = []
                    for item in top_10:
                        row = {
                            "similitud_coseno": item.get("similitud"),
                            "indice_relativo": item.get("indice_relativo"),
                            "indice_global": item.get("indice_global"),
                            "spatial": item["vector_original"].get("spatial") if "vector_original" in item else None,
                            "temporal": item["vector_original"].get("temporal") if "vector_original" in item else None,
                            "interest": item["vector_original"].get("interest") if "vector_original" in item else None,
                            "reference": item["vector_original"].get("reference") if "vector_original" in item else None,
                            "observation": item["vector_original"].get("observation") if "vector_original" in item else None,
                            "sentencia_procesada": item.get("sentencia_procesada"),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        rows.append(row)

                    df_similitudes = pd.DataFrame(rows)
                    df_similitudes.to_csv(csv_path, index=False)
                    response["similitudes_csv"] = csv_path
                except Exception as e:
                    response["top_10_error"] = str(e)
                except FileNotFoundError:
                    response["top_10_error"] = "No se pudo cargar el CSV original '../data/sample.csv'"
                except Exception as e:
                    response["top_10_error"] = f"Error al cargar CSV original: {str(e)}"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
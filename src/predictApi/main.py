from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from src.Modules.predict_vector import PredictVector
from src.Modules.classification_manager import ClassificationManager
import unicodedata
import os
import json
import time
import datetime

app = FastAPI()
classifier_manager = ClassificationManager(random_state=42)
dataset_originales_path = "./data/sample.csv"

DS = "DS2"
    
class TrainRequest(BaseModel):
    modelo: str = "st1"
    params: str = "separate_grid"
    models_dir: str = "test/Modelos"
    ds_originales_path: str = "./data/sample.csv"
    name_modelo: str = "mlp"
    use_adjusted: bool = False
    embeddings_path: str = "test/Embeddings"

# Estructura para manejar múltiples clasificadores cargados
loaded_classifiers = {}

def add_loaded_classifier(model_key: str, classifier_manager:dict):
    """Añade un clasificador cargado al registro JSON de clasificadores"""
    loaded_classifiers_registry = {}
    try:
        # Actualizar el diccionario en memoria
        loaded_classifiers[model_key] = classifier_manager
            
        print(f"Clasificador añadido al registro: {model_key}")
        return "Clasificador añadido al registro"
    except Exception as e:
        print(f"Error al añadir clasificador al registro: {str(e)}")

    
def get_model_key(request: TrainRequest):
    """Genera una clave única para identificar un modelo"""
    adjusted_suffix = "_adjusted" if request.use_adjusted else ""
    return f"{request.modelo}_{request.params}{adjusted_suffix}"

@app.get("/listClassifiers")
def list_classifiers():
    """Retorna la lista de clasificadores cargados sin el modelo en memoria"""
    try:
        classifiers_info = {}
        for model_key, classifier_data in loaded_classifiers.items():
            # Crear una copia de los datos sin el classifier_model
            classifier_info = {
            "modelo_name": classifier_data.get("modelo_name"),
            "originales_path": classifier_data.get("originales_path"),
            "use_adjusted": classifier_data.get("use_adjusted"),
            "classifier_type": classifier_data.get("classifier_type"),
            "embeddings_path": classifier_data.get("embeddings_path")
            }
            classifiers_info[model_key] = classifier_info
        
        return {
            "total_models": len(classifiers_info),
            "loaded_classifiers": classifiers_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/loadClassifier")
def load_classifier(request: TrainRequest):
    try:
        # Ejemplo de JSON para el request:
        # {
        #     "modelo": "st1",
        #     "params": "separate_grid",
        #     "models_dir": "test/Modelos",
        #     "ds_originales_path": "./data/sample.csv",
        #     "name_modelo": "mlp",
        #     "use_adjusted": false,
        #     "embeddings_path": "test/Embeddings"
        # }
        dataset_originales_path = request.ds_originales_path
        # Aquí iría la lógica para entrenar el clasificador
        # Cargar datos de clustering para entrenar el clasificador
        print("Iniciando carga del clasificador...")
        print(f"Parámetros recibidos: {request}")

        # Aquí iría la lógica para entrenar el clasificador
        # print("Cargando datos de clustering para entrenar el clasificador...")

        model_suffix = "_adjusted" if request.use_adjusted == True else  ""
        model_params = request.params
        if request.params == "separate_grid":
            model_params = "gridSearch"
        
        # Crear directorio de modelos si no existe
        model_name = request.name_modelo
        print("--------------------------------", model_name)
        model_dir = os.path.join(request.models_dir + model_suffix, request.modelo, f"classifier_{model_name}{model_suffix}.pkl")
        print(model_dir)
        # print(f"Ruta de embeddings etiquetados: {embeddings_originales_labeled_path}")
        model_loaded = classifier_manager.load_model(
            model_path=model_dir
        )
        
        classifier_add = {
            "modelo_name": request.modelo,
            "originales_path": dataset_originales_path,
            "use_adjusted": model_suffix,
            "classifier_model": model_loaded,
            "classifier_type": model_name,
            "embeddings_path": request.models_dir,
            "params": model_params,
            "prefix_params": request.params
        }
        
        name_model_add = f"{request.modelo}_{model_name}{model_suffix}"

        model_registred = add_loaded_classifier(name_model_add, classifier_add)

        print(f"Modelo cargado: {model_loaded}", "\t Tipo:", type(model_loaded))
        
        return {"message": f"Clasificador cargado exitosamente. Modelo: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def limpiar_texto(texto):
    if isinstance(texto, list):
        texto = ' '.join([str(x) for x in texto])
    # # Convierte a minúsculas
    texto = texto.lower()
    # # Elimina acentos
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    # # Reemplaza espacios por guiones bajos
    texto = texto.replace(' ', '_')
    # # Elimina comas
    texto = texto.replace(',', '')
    return texto

def save_prediction_log(query_string: str, predicted_label: int, confidence: float, predict: str, modelo: str, clasificador: str, use_adjusted: bool, start_time: float, end_time: float, cant_group: int, total_embeddings: int, csv_path: str = "./prediction_log.csv"):
    """
    Guarda el log de predicciones en un CSV con contador automático
    """
    try:
        # Crear el registro de la predicción
        prediction_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query_original": query_string,
            "grupo_predicho": predicted_label,
            "confianza": confidence,
            "prediccion": predict,
            "modelo": modelo,
            "clasificador": clasificador,
            "use_adjusted": use_adjusted,
            "start_time": start_time,
            "end_time": end_time,
            "tiempo_consulta_segundos": round(end_time - start_time, 2),
            "cant_group": cant_group,
            "total_embeddings": total_embeddings
        }

        # Verificar si el archivo existe
        if os.path.exists(csv_path):
            # Cargar el CSV existente
            existing_df = pd.read_csv(csv_path)
            # Obtener el próximo contador
            next_counter = existing_df['id'].max() + 1 if not existing_df.empty else 1
        else:
            # Si no existe, comenzar el contador en 1
            next_counter = 1
            existing_df = pd.DataFrame()
        
        # Añadir el contador al registro
        prediction_record["id"] = next_counter
        
        # Crear DataFrame con el nuevo registro
        new_record_df = pd.DataFrame([prediction_record])
        
        # Combinar con datos existentes si los hay
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        else:
            combined_df = new_record_df
        
        # Guardar el CSV actualizado
        combined_df.to_csv(csv_path, index=False)
        print(f"Predicción guardada en log: {csv_path} - Contador: {next_counter}")
        
        return next_counter
        
    except Exception as e:
        print(f"Error al guardar log de predicción: {str(e)}")
        return None



# Modelo de entrada para la API
class PredictRequest(BaseModel):
    modelo: str = "st1"
    classifier_model: str = "mlp"
    use_adjusted: bool = False
    vector_input: list

# Estructura para manejar modelos de sentence transformer cargados
loaded_sentence_transformers = {}

def load_sentence_transformer_model(model_name: str, model_type: str, model_path: str):
    """Carga un modelo de sentence transformer y lo mantiene en memoria"""
    try:
        if model_name not in loaded_sentence_transformers:
            print(f"Cargando modelo de sentence transformer: {model_name}")
            predictor = PredictVector(model_name, model_type, model_path)
            loaded_sentence_transformers[model_name] = predictor
            print(f"Modelo {model_name} cargado exitosamente en memoria")
        else:
            print(f"Modelo {model_name} ya está cargado en memoria")
        return loaded_sentence_transformers[model_name]
    except Exception as e:
        print(f"Error al cargar modelo {model_name}: {str(e)}")
        raise e

def get_sentence_transformer_model(model_name: str):
    """Obtiene un modelo de sentence transformer de la memoria"""
    if model_name in loaded_sentence_transformers:
        return loaded_sentence_transformers[model_name]
    else:
        # Si no está cargado, cargarlo automáticamente
        modelos_dict = {
            "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
            "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
            "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
        }
            
        if model_name in modelos_dict:
            nombre, tipo, ruta = modelos_dict[model_name]
            return load_sentence_transformer_model(nombre, tipo, ruta)
        else:
            raise ValueError(f"Modelo no reconocido: {model_name}")

@app.on_event("startup")
def startup_event():
    """Carga automáticamente todos los modelos de sentence transformer al iniciar la aplicación"""
    modelos_dict = {
        "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
    }
    
    print("Cargando modelos de sentence transformer automáticamente...")
    for model_key, (nombre, tipo, ruta) in modelos_dict.items():
        try:
            load_sentence_transformer_model(nombre, tipo, ruta)
            print(f"✓ Modelo {model_key} cargado exitosamente")
        except Exception as e:
            print(f"✗ Error cargando modelo {model_key}: {str(e)}")

@app.get("/listSentenceTransformers")
def list_sentence_transformers():
    """Lista los modelos de sentence transformer cargados en memoria"""
    return {
            "total_models": len(loaded_sentence_transformers),
            "loaded_models": list(loaded_sentence_transformers.keys())
        }
  
@app.post("/predict")
def predict(request: PredictRequest):
    # Ejemplo de JSON de entrada para /predict:
    # {
    #     "modelo": "st1",                # Modelo de embeddings a usar: "st1", "st2", "st3" o "use"
    #     "classifier_model": "mlp",      # Nombre del modelo de clasificador cargado
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
        start_time = time.time()
        model_suffix = "_adjusted" if request.use_adjusted == True else  ""
        # Preparar vector de entrada
        vector_input = request.vector_input
        if request.use_adjusted:
            spatial = vector_input[0]
            temporal = vector_input[1]
            interest = vector_input[2]
            reference = vector_input[3]
            observable = vector_input[4]
            vector_input = [spatial, temporal, interest]
        
        # query_string = limpiar_texto(vector_input)
        query_string = ' '.join([str(x).replace(' ', '_') for x in vector_input])
        print(f"Vector de entrada procesado: {query_string}")
        # Selección del modelo de embeddings
        # modelos_dict = {
        #     "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        #     "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        #     "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
        # }
        predictor = loaded_sentence_transformers[request.modelo]
        print(f"Modelo seleccionado: {request.modelo}")

        # Generar embedding
        embedding = predictor.generar_embedding(query_string)
        print(f"Embedding generado tamaño de: {len(embedding)}")
        # best_model = classifier_manager.best_model
        # print(f"Prediccion {classifier_manager.best_model}")
        
        # carga del modelo en el json
        model_key = f"{request.modelo}_{request.classifier_model}{model_suffix}"
        print("Modelo a usar:", model_key)
        if model_key not in loaded_classifiers:
            raise HTTPException(status_code=400, detail=f"Clasificador no cargado: {model_key}. Cargue el clasificador primero.")
        else:
            classifier_model = loaded_classifiers[model_key]
            print(f"Clasificador encontrado en el registro: {model_key}")
        dataset_originales_path = classifier_model["originales_path"]
        predictions, probabilities = classifier_manager.predict_query(
            model=classifier_model["classifier_model"],
            new_embeddings=embedding.reshape(1, -1)
        )
        predicted_label = predictions[0]
        
        # Guardar embedding en CSV
        try:
            embedding_row = {
                "vector_input": str(vector_input),
                "query_string": query_string,
                "embedding": embedding.tolist(),
                "predicted_label": int(predicted_label) if predicted_label is not False else False,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Definir nombre del archivo CSV para embeddings
            embedding_csv_dir = "./"
            embedding_csv_path = os.path.join(embedding_csv_dir, f"embedding_generado_{DS}.csv")
            
            # Si el archivo existe, cargar y agregar; si no, crear nuevo
            if os.path.exists(embedding_csv_path):
                existing_embeddings_df = pd.read_csv(embedding_csv_path)
                new_embedding_df = pd.DataFrame([embedding_row])
                combined_df = pd.concat([existing_embeddings_df, new_embedding_df], ignore_index=True)
            else:
                combined_df = pd.DataFrame([embedding_row])
            
            combined_df.to_csv(embedding_csv_path, index=False)
            print(f"Embedding guardado en: {embedding_csv_path}")
            
        except Exception as e:
            print(f"Error al guardar embedding: {str(e)}")
        
        
        
        response = {
            "grupo_predicho": int(predicted_label) if predicted_label is not False else False,
            "modelo": request.modelo,
            "clasificador": str(request.classifier_model),
        }
        print("Respuesta generada:\n" + json.dumps(response, indent=4, ensure_ascii=False))
        # Cargar embeddings etiquetados para análisis adicional
        path_embeddings_labeled = os.path.join(
            f"{classifier_model['embeddings_path']}_{classifier_model['modelo_name']}{classifier_model['use_adjusted']}",
            classifier_model['params'],
            "HDBSCAN",
            f"embeddings_labeled_{classifier_model['prefix_params']}.csv"
        )
        print("Cargando embeddings etiquetados desde:", path_embeddings_labeled)
        embeddings, labels = classifier_manager.load_data(path_embeddings_labeled)
        # labels = classifier_manager.labels
        # embeddings = classifier_manager.embeddings
   
   
        if probabilities is not None:
            max_prob = float(np.max(probabilities[0]))
            response["confianza"] = max_prob
            unique_labels = np.unique(labels)
            probabilities_list = probabilities[0].tolist()

        print(f"Grupo predicho: {predicted_label} con confianza {response.get('confianza', 'N/A')}")
        
        
        
        top_idx = 10
        
        # Verificar si el embedding ya existe exactamente en el grupo
        existe, idx_relativo, idx_global, size_group, similarities  = predictor.buscar_similares_en_grupo(
            embedding=embedding,
            embeddings_labeled_path=path_embeddings_labeled,
            grupo_id=predicted_label,
            top_n=top_idx
        )
        
        
        print(f"¿Existe en el grupo {predicted_label}? {existe}")
        response["existe_en_grupo"] = existe
        print("Cargando CSV original desde:", dataset_originales_path)
        if not os.path.exists(dataset_originales_path):
            raise FileNotFoundError(f"El archivo CSV no existe: {dataset_originales_path}")
        else:
            print(f"El archivo CSV existe: {dataset_originales_path}")
            csv_orig = pd.read_csv(dataset_originales_path, encoding='utf-8')
        print(csv_orig.head())
                    
        if existe:
            print(f"El embedding ya existe en el grupo: {predicted_label}")
            response["indice_relativo"] = int(idx_relativo)
            response["indice_global"] = int(idx_global)
            # Opcional: cargar vector original si se requiere
            try:
                # df_embeddings = pd.read_csv(dataset_originales_path)
                if existe and idx_global is not None:
                    vector_original = csv_orig.iloc[int(idx_global)].to_dict()
                    response["vector_original"] = vector_original

            except Exception as e:
                response["vector_original_error"] = str(e)
        else:
            # Buscar similares dentro del grupo predicho
            # top_idx, similarities, idx_global, embeddings_group = predictor.buscar_similares_en_grupo_por_etiqueta(
            #     embedding=embedding,
            #     embeddings_labeled_path=path_embeddings_labeled,
            #     grupo_id=predicted_label,
            #     top_n=10
            # )
            # print(f"Top 10 índices relativos en el grupo: {top_idx}")
            if top_idx > 0:
                # Cargar el CSV original para extraer los vectores originales
                print(f"Tamaño de similitudes: {len(similarities)}, indices: {len(idx_relativo)}")
                try:
                    top_10 = []
                    for i, (rel_idx, glob_idx, sim) in enumerate(zip(idx_relativo, idx_global, similarities)):
                        print(f"Procesando similaridad {i+1}: Índice relativo {rel_idx}, Índice global {glob_idx}, Similitud {sim}")
                        if glob_idx < csv_orig.shape[0]:
                            print(f"Índice relativo: {rel_idx}, Índice global: {glob_idx}, Similitud: {sim}")
                            vector_original = csv_orig.iloc[glob_idx]
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
                    csv_base = f"similitudes_consulta_{request.modelo}_{request.classifier_model}{model_suffix}"
                    csv_dir = f"./similitudes/{request.modelo}_{request.classifier_model}{model_suffix}"
                    consulta_num = 1

                    # Buscar un nombre de archivo que no exista aún
                    while os.path.exists(os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")):
                        consulta_num += 1
                    # Crear directorio si no existe
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_path = os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")
                    print(f"Guardando similitudes en: {csv_path}")
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
                    print(f"Similitudes guardadas en: {csv_path}")
                    response["similitudes_csv"] = str(csv_path)
                except Exception as e:
                    response["top_10_error"] = str(e)
                except FileNotFoundError:
                    response["top_10_error"] = "No se pudo cargar el CSV original '../data/sample.csv'"
                except Exception as e:
                    response["top_10_error"] = f"Error al cargar CSV original: {str(e)}"
        
        
        
        # Llamar a la función para guardar el log
        porcentaje_reduccion = size_group / len(labels) * 100 if size_group > 0 else 0
        prediction_counter = save_prediction_log(
            query_string=query_string,
            predicted_label=predicted_label,
            confidence=response.get('confianza', 0.0),
            predict=existe,
            modelo=request.modelo,
            clasificador=request.classifier_model,
            use_adjusted=bool(request.use_adjusted),
            csv_path=f"./prediction_{DS}_log.csv",
            start_time=start_time,
            end_time=time.time(),
            cant_group=size_group,
            total_embeddings=len(labels)
        )
        
        return existe

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






def save_prediction_log_all(query_string: str, confidence: float, predict: str, modelo: str, use_adjusted: bool, start_time: float, end_time: float, total_embeddings: int, csv_path: str = "./prediction_log.csv"):
    """
    Guarda el log de predicciones en un CSV con contador automático
    """
    try:
        # Crear el registro de la predicción
        prediction_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query_original": query_string,
            "confianza": confidence,
            "prediccion": predict,
            "modelo": modelo,
            "use_adjusted": use_adjusted,
            "start_time": start_time,
            "end_time": end_time,
            "tiempo_consulta_segundos": round(end_time - start_time, 2),
            "total_embeddings": total_embeddings
        }

        # Verificar si el archivo existe
        if os.path.exists(csv_path):
            # Cargar el CSV existente
            existing_df = pd.read_csv(csv_path)
            # Obtener el próximo contador
            next_counter = existing_df['id'].max() + 1 if not existing_df.empty else 1
        else:
            # Si no existe, comenzar el contador en 1
            next_counter = 1
            existing_df = pd.DataFrame()
        
        # Añadir el contador al registro
        prediction_record["id"] = next_counter
        
        # Crear DataFrame con el nuevo registro
        new_record_df = pd.DataFrame([prediction_record])
        
        # Combinar con datos existentes si los hay
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        else:
            combined_df = new_record_df
        
        # Guardar el CSV actualizado
        combined_df.to_csv(csv_path, index=False)
        print(f"Predicción guardada en log: {csv_path} - Contador: {next_counter}")
        
        return next_counter
        
    except Exception as e:
        print(f"Error al guardar log de predicción: {str(e)}")
        return None


@app.post("/predict/all")
def predict_all(request: PredictRequest):
    # Ejemplo de JSON de entrada para /predict:
    # {
    #     "modelo": "st1",                # Modelo de embeddings a usar: "st1", "st2", "st3" o "use"
    #     "classifier_model": "mlp",      # Nombre del modelo de clasificador cargado
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
        start_time = time.time()
        model_suffix = "_adjusted" if request.use_adjusted == True else  ""
        # Preparar vector de entrada
        vector_input = request.vector_input
        if request.use_adjusted:
            spatial = vector_input[0]
            temporal = vector_input[1]
            interest = vector_input[2]
            reference = vector_input[3]
            observable = vector_input[4]
            vector_input = [spatial, temporal, interest]
        
        # query_string = limpiar_texto(vector_input)
        query_string = ' '.join([str(x).replace(' ', '_') for x in vector_input])
        print(f"Vector de entrada procesado: {query_string}")
        # Selección del modelo de embeddings
        # modelos_dict = {
        #     "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
        #     "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
        #     "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
        # }
        predictor = loaded_sentence_transformers[request.modelo]
        print(f"Modelo seleccionado: {request.modelo}")

        # Generar embedding
        embedding = predictor.generar_embedding(query_string)
        print(f"Embedding generado tamaño de: {len(embedding)}")
        # best_model = classifier_manager.best_model
        # print(f"Prediccion {classifier_manager.best_model}")
        
        # carga del modelo en el json
        model_key = f"{request.modelo}_{request.classifier_model}{model_suffix}"
        print("Modelo a usar:", model_key)
        if model_key not in loaded_classifiers:
            raise HTTPException(status_code=400, detail=f"Clasificador no cargado: {model_key}. Cargue el clasificador primero.")
        else:
            classifier_model = loaded_classifiers[model_key]
            print(f"Clasificador encontrado en el registro: {model_key}")
        dataset_originales_path = classifier_model["originales_path"]
        
        
        # Guardar embedding en CSV
        try:
            embedding_row = {
                "vector_input": str(vector_input),
                "query_string": query_string,
                "embedding": embedding.tolist(),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Definir nombre del archivo CSV para embeddings
            embedding_csv_dir = "./"
            embedding_csv_path = os.path.join(embedding_csv_dir, f"embedding_generado_all_{DS}.csv")
            
            # Si el archivo existe, cargar y agregar; si no, crear nuevo
            if os.path.exists(embedding_csv_path):
                existing_embeddings_df = pd.read_csv(embedding_csv_path)
                new_embedding_df = pd.DataFrame([embedding_row])
                combined_df = pd.concat([existing_embeddings_df, new_embedding_df], ignore_index=True)
            else:
                combined_df = pd.DataFrame([embedding_row])
            
            combined_df.to_csv(embedding_csv_path, index=False)
            print(f"Embedding guardado en: {embedding_csv_path}")
            
        except Exception as e:
            print(f"Error al guardar embedding: {str(e)}")
        
        
        # Cargar embeddings etiquetados para análisis adicional
        path_embeddings_labeled = os.path.join(
            f"{classifier_model['embeddings_path']}_{classifier_model['modelo_name']}{classifier_model['use_adjusted']}",
            classifier_model['params'],
            "HDBSCAN",
            f"embeddings_labeled_{classifier_model['prefix_params']}.csv"
        )
        print("Cargando embeddings etiquetados desde:", path_embeddings_labeled)
        embeddings, labels = classifier_manager.load_data(path_embeddings_labeled)
        # labels = classifier_manager.labels
        # embeddings = classifier_manager.embeddings
        
        top_idx = 10
        
        # Verificar si el embedding ya existe exactamente en el grupo
        existe, idx_exacto, idx_top, similitudes = predictor.buscar_en_total_con_top_similares(
            embedding_query=embedding, 
            embeddings_labeled_path=path_embeddings_labeled,
            top_n=top_idx
        )

        
        print("Cargando CSV original desde:", dataset_originales_path)
        if not os.path.exists(dataset_originales_path):
            raise FileNotFoundError(f"El archivo CSV no existe: {dataset_originales_path}")
        else:
            print(f"El archivo CSV existe: {dataset_originales_path}")
            csv_orig = pd.read_csv(dataset_originales_path, encoding='utf-8')
        # print(csv_orig.head())
                    
                    
        response={"existe_en_grupo": existe}
        print(f"¿Existe en el csv ? {existe}")
        
        if existe:
            print(f"El embedding ya existe en el csv")
            response["indice_global"] = int(idx_exacto)
            # Opcional: cargar vector original si se requiere
            try:
                # df_embeddings = pd.read_csv(dataset_originales_path)
                csv_orig = pd.read_csv(dataset_originales_path, encoding='utf-8')
                if existe and idx_exacto is not None:
                    vector_original = csv_orig.iloc[int(idx_exacto)].to_dict()
                    response["vector_original"] = vector_original

            except Exception as e:
                response["vector_original_error"] = str(e)
        else:
            # Buscar similares dentro del grupo predicho

            print(f"Top {top_idx} índices globales")
            if top_idx > 0:
                # Cargar el CSV original para extraer los vectores originales
                print(f"Indices: {(top_idx)}")
                try:
                    
                    top_10 = []
                    csv_orig = pd.read_csv(dataset_originales_path, encoding='utf-8')
                    # print(f"Cargado CSV original desde: {dataset_originales_path}")
                    print("similaridades:", similitudes)
                    for i, (glob_idx, similitud) in enumerate(zip(idx_top, similitudes)):
                        if glob_idx < len(csv_orig):
                            vector_original = csv_orig.iloc[glob_idx]
                            if request.use_adjusted:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']}"
                            else:
                                sentencia_procesada = f"{vector_original['spatial']} {vector_original['temporal']} {vector_original['interest']} {vector_original['reference']} {vector_original['observation']}"
                            top_10.append({
                                "similitud": float(similitud),
                                "indice_relativo": i,
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
                                "indice_relativo": i,
                                "indice_global": int(glob_idx),
                                "error": f"Índice global {glob_idx} fuera del rango del CSV original"
                            })
                    response["top_10"] = top_10

                    # Definir nombre base del archivo
                    csv_base = f"similitudes_consulta_{request.modelo}_{request.classifier_model}{model_suffix}"
                    csv_dir = f"./similitudes_all/{request.modelo}_{request.classifier_model}{model_suffix}"
                    consulta_num = 1

                    # Buscar un nombre de archivo que no exista aún
                    while os.path.exists(os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")):
                        consulta_num += 1
                    # Crear directorio si no existe
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_path = os.path.join(csv_dir, f"{csv_base}_{consulta_num}.csv")
                    print(f"Guardando similitudes en: {csv_path}")
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
                    print(f"Similitudes guardadas en: {csv_path}")
                    response["similitudes_csv"] = str(csv_path)
                except Exception as e:
                    response["top_10_error"] = str(e)
                except FileNotFoundError:
                    response["top_10_error"] = "No se pudo cargar el CSV original '../data/sample.csv'"
                except Exception as e:
                    response["top_10_error"] = f"Error al cargar CSV original: {str(e)}"
        
        
        
        # Llamar a la función para guardar el log
        prediction_counter = save_prediction_log_all(
            query_string=query_string,
            confidence=response.get('confianza', 0.0),
            predict=existe,
            modelo=request.modelo,
            use_adjusted=bool(request.use_adjusted),
            csv_path=f"./prediction_all_{DS}_log.csv",
            start_time=start_time,
            end_time=time.time(),
            total_embeddings=len(labels)
        )
        
        return existe

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
       
    
    
import requests
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import time
# Mezclar aleatoriamente la columna 'interest' de un CSV
import random


class TrainRequest(BaseModel):
    modelo: str = "st1"
    params: str = "separate_grid"
    models_dir: str = "test/Modelos"
    ds_originales_path: str = "./data/sample.csv"
    name_modelo: str = "mlp"
    use_adjusted: bool = False
    embeddings_path: str = "test/Embeddings"


class PredictRequest(BaseModel):
    modelo: str = "st1"
    classifier_model: str = "mlp"
    use_adjusted: bool = False
    vector_input: list

def send_train_request(url: str, request_data: Optional[TrainRequest | PredictRequest] = None):
    """Envía una petición de entrenamiento al servidor"""
    
    # Usar valores por defecto si no se proporciona request_data
    if request_data is None:
        request_data = TrainRequest()
    
    # Convertir el modelo Pydantic a dict
    json_data = request_data.dict()
    
    try:
        response = requests.post(
            url,
            json=json_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error en la petición: {e}")
        return None



# Ejemplo de uso
if __name__ == "__main__":
    
    # URL del endpoint
    api_url = "http://localhost:8000/loadClassifier"  # Cambia por tu URL
    
    model_embedding = ["st1", "st2", "st3"]
    params = ["separate_grid"]
    class_model = ["mlp", "random_forest", "svm"]
    
    # Crear petición con valores personalizados
    
    # {
    #     "modelo": "st1",
    #     "params": "separate_grid",
    #     "models_dir": "./test_DS1/Modelos",
    #     "ds_originales_path": "./data/sample.csv",
    #     "name": "mlp",
    #     "use_adjusted": "True",
    #     "embeddings_path": "./testDS1"
    # }
    
    DS = "DS2"
    for m_e in model_embedding:
        for c_m in class_model:
            custom_request = TrainRequest(
                    modelo=m_e,
                    params="separate_grid",
                    models_dir=f"./test_{DS}/Modelos",
                    ds_originales_path=f"./data/sample_{DS.lower()}.csv",
                    name_modelo=c_m,
                    use_adjusted=True,
                    embeddings_path=f"./test{DS}"
                )

        # Enviar petición
            result = send_train_request(api_url, custom_request)
        # result = "--- IGNORE ---"
            if result:
                print("Respuesta del servidor:", result)
            else:
                print("Error al procesar la petición")
            
    # exit(0)
    # Leer el archivo CSV
    # exit(0)
    df = pd.read_csv(f"no_exact_test_{DS}.csv")

    print(df.head())
    
    #{
    #   "modelo": "st1",
    #   "classifier_model": "mlp",
    #   "use_adjusted": "True",
    #   "vector_input": [
    #       "coahuila_de_zaragoza.escobedo",
    #       "2020",
    #       "total.total",
    #       "0.294290759",
    #       "-"
    #   ]
    # }
    api_url = "http://localhost:8000/predict/all"  # Cambia por tu URL
    for m_e in model_embedding:
        for index, row in df.iterrows():
            for c_m in class_model:
                custom_request = PredictRequest(
                    modelo=m_e,
                    classifier_model=c_m,
                    use_adjusted=True,
                    vector_input=[str(x) for x in row.tolist()]
                )
                # Enviar petición
                print("id=", index," model =", m_e, "classifier =", c_m)
                result = send_train_request(api_url, custom_request)
                # result = "--- IGNORE ---"
                if result:
                    print("Respuesta del servidor:", result, "para el input:", custom_request.vector_input)
                else:
                    print("Error al procesar la petición")
                    # time.sleep(5)

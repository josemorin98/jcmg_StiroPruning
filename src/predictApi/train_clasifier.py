import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
from src.Modules.classification_manager import ClassificationManager

classifier_manager = ClassificationManager(random_state=42)

def train_and_save_classifiers(request: dict):
    # {
        #     "modelo": "st1",
        #     "params": "bayesian",
        #     "embeddings_path": "test/Embeddings",
        #     "models_dir": "test/Modelos",
        #     "ds_originales_path": "../data/sample.csv",
        #     "use_adjusted": false
    # }
    
    print("Iniciando entrenamiento del clasificador...")
    print(f"Parámetros recibidos: {request}")

    # Aquí iría la lógica para entrenar el clasificador
    print("Cargando datos de clustering para entrenar el clasificador...")

    model_suffix = "_adjusted" if request["use_adjusted"] == True else  ""
    params_suffix = f"_{request['params']}"
    model_params = request['params']
    if request['params'] == "separate_grid":
        model_params = "gridSearch"
        
    # Crear directorio de modelos si no existe
    os.makedirs(f"{request['models_dir']}_{request['modelo']}{model_suffix}/{model_params}/HDBSCAN", exist_ok=True)
    models_dir = f"{request['models_dir']}_{request['modelo']}{model_suffix}"
    embeddings_originales_labeled_path = os.path.abspath(f"{models_dir}/{model_params}/HDBSCAN/embeddings_labeled_{request['params']}.csv")

        
    # print(f"Ruta de embeddings etiquetados: {embeddings_originales_labeled_path}")
    embeddings_data, labels = classifier_manager.load_data(
            embeddings_path=embeddings_originales_labeled_path
        )

    # Entrenar clasificadores
    results = classifier_manager.train_classifiers(embeddings_data, labels, cv_folds=3)
    classifier_manager.save_models(modelos_dir=f"{request['models_dir']}{model_suffix}/{request['modelo']}",
                                   model_suffix=model_suffix)
    # Guardar resultados en formato requerido
    results_list = []
    for model_name, metrics in results.items():
            results_list.append({
                "embeding_model": request['modelo'],
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
    
    return results

def load_model(model_name:str, models_dir:str="models"):
    """
    Carga un modelo previamente entrenado
    
    Args:
        model_name (str): Nombre del modelo a cargar
        models_dir (str): Directorio donde están los modelos
    
    Returns:
        modelo entrenado
    """
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Modelo {model_name} no encontrado en {model_path}")

if __name__ == "__main__":
    
    # {
        #     "modelo": "st1",
        #     "params": "bayesian",
        #     "embeddings_path": "test/Embeddings",
        #     "models_dir": "test/Modelos",
        #     "ds_originales_path": "../data/sample.csv",
        #     "use_adjusted": false
    # }
    
    modelo = ["st1", "st2", "st3"]
    params = "separate_grid"
    
    for m in modelo:
        requet = {
            "modelo": m,
            "params": params,
            "embeddings_path": "test/Embeddings",
            "models_dir": "test/Modelos",
            "ds_originales_path": "../data/sample.csv",
            "use_adjusted": True
        }
        
        results = train_and_save_classifiers(requet)
        print(f"Resultados para modelo {m}: {results}")
    
    # Mostrar resultados
    print("\n Modelos entrenados")
import pandas as pd
from Modules.model_manager import EmbeddingModelManager
from hyperopt import hp
import time
import pickle
import os
import argparse

path_test  = "../test"


# -------------------------------------------------------------------------------
# Argumentos de línea de comandos para seleccionar el modelo
parser = argparse.ArgumentParser(description="Generar embeddings con el modelo seleccionado.")
parser.add_argument(
    "--modelo",
    type=str,
    choices=["use", "st1", "st2", "st3"],
    required=True,
    help="Modelo de embeddings a utilizar: use, st1, st2 o st3"
)
args = parser.parse_args()

# -------------------------------------------------------------------------------
# Cargar el dataset de intents
print("Cargando dataset...")
start_time = time.time()
data_sample = pd.read_csv("../data/sample.csv")
tiempo_carga = time.time() - start_time
print(f"Dataset cargado en {tiempo_carga:.2f} segundos.")

# Concatenar todas las sentencias separadas por un espacio
print("\nProcesando texto: reemplazando espacios y concatenando columnas...")
start_time = time.time()
all_intents = data_sample.astype(str).applymap(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()
tiempo_procesado = time.time() - start_time
print(f"Procesamiento de texto completado en {tiempo_procesado:.2f} segundos.")

print(f"\nTotal de registros procesados: {len(all_intents)}")
print("Ejemplo de texto procesado:")
print(all_intents[0])

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
print(f"\nGenerando y guardando embeddings para {nombre}...")
start_time = time.time()
embeddings = manager.embed(nombre, all_intents, save=True, save_csv=True)
tiempo_embeddings = time.time() - start_time
print(f"Embeddings generados y guardados en {tiempo_embeddings:.2f} segundos.")

# Guardar los tiempos en un CSV
times = pd.DataFrame([{
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "modelo": nombre,
    "tiempo_carga_dataset": round(tiempo_carga, 2),
    "tiempo_procesado_texto": round(tiempo_procesado, 2),
    "tiempo_carga_modelo": round(tiempo_carga_modelo, 2),
    "tiempo_generacion_embeddings": round(tiempo_embeddings, 2)
}])

csv_path = f"{path_test}/embeddings/times_embeddings.csv"
os.makedirs(f"{path_test}/embeddings", exist_ok=True)

if os.path.exists(csv_path):
    times_ant = pd.read_csv(csv_path)
    times = pd.concat([times_ant, times], ignore_index=True)

times.to_csv(csv_path, index=False)
print(f"\nTiempos guardados en '{csv_path}'")
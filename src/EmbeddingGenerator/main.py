from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import time
import asyncio
from datetime import datetime
import sys
import uvicorn

# Agregar el directorio padre al path para importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.model_manager import EmbeddingModelManager
from Modules.predict_vector import PredictVector
from Modules.classification_manager import ClassificationManager

# Configuración
app = FastAPI(
    title="JCMG - Embedding Generator API",
    description="API para generar embeddings usando diferentes modelos de transformers",
    version="1.0.0"
)

# Modelos disponibles
MODELOS_DISPONIBLES = {
    "use": ("use", "use", "https://tfhub.dev/google/universal-sentence-encoder/4"),
    "st1": ("st1", "sentence_transformer", "all-mpnet-base-v2"),
    "st2": ("st2", "sentence_transformer", "all-MiniLM-L6-v2"),
    "st3": ("st3", "sentence_transformer", "paraphrase-mpnet-base-v2")
}

# Variables globales
manager = None
modelos_cargados = {}
path_test = "test"

# Modelos Pydantic
class PredictRequest(BaseModel):
    texto: str
    modelo: str = "st1"
    params: str = "bayesian"
    use_adjusted: bool = False
    embeddings_labeled_path: Optional[str] = None

class PredictResponse(BaseModel):
    predicted_label: int
    similarity_score: float
    confidence: Optional[float] = None
    embedding_dimensions: int
    modelo_usado: str
    existe_exacto: bool
    idx_relativo: Optional[int] = None
    idx_global: Optional[int] = None
    similares_en_grupo: List[Dict[str, Any]] = []

class TrainClassifierRequest(BaseModel):
    embeddings_labeled_path: str
    modelo: str = "st1"
    params: str = "bayesian"
    use_adjusted: bool = False

class TrainClassifierResponse(BaseModel):
    best_model: str
    best_score: float
    training_time: float
    num_classes: int
    num_samples: int

class GenerarDatasetRequest(BaseModel):
    archivo_csv: str = "../data/sample.csv"
    modelo: str = "st1"
    columnas: List[str] = ["Spatial", "Temporal", "Interest"]
    incluir_reference_observation: bool = True

class StatusResponse(BaseModel):
    estado: str
    mensaje: str
    modelos_cargados: List[str]
    tiempo_activo: str

# Funciones auxiliares
def inicializar_manager():
    """Inicializa el manager de embeddings"""
    global manager
    if manager is None:
        manager = EmbeddingModelManager(save_dir=f"{path_test}/embeddings")
    return manager

def cargar_modelo(nombre_modelo: str):
    """Carga un modelo específico si no está ya cargado"""
    global modelos_cargados, manager
    
    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=400, detail=f"Modelo {nombre_modelo} no disponible")
    
    if nombre_modelo not in modelos_cargados:
        manager = inicializar_manager()
        nombre, tipo, ruta = MODELOS_DISPONIBLES[nombre_modelo]
        
        print(f"Cargando modelo {nombre}...")
        start_time = time.time()
        manager.load_model(nombre, tipo, ruta)
        tiempo_carga = time.time() - start_time
        
        modelos_cargados[nombre_modelo] = {
            "nombre": nombre,
            "tipo": tipo,
            "tiempo_carga": tiempo_carga,
            "timestamp": datetime.now().isoformat()
        }
        print(f"Modelo {nombre} cargado en {tiempo_carga:.2f} segundos")
    
    return modelos_cargados[nombre_modelo]

# Endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Endpoint raíz con información del servicio"""
    return StatusResponse(
        estado="activo",
        mensaje="JCMG Embedding Generator API funcionando correctamente",
        modelos_cargados=list(modelos_cargados.keys()),
        tiempo_activo=datetime.now().isoformat()
    )

@app.get("/modelos")
async def listar_modelos():
    """Lista todos los modelos disponibles"""
    return {
        "modelos_disponibles": list(MODELOS_DISPONIBLES.keys()),
        "modelos_cargados": list(modelos_cargados.keys()),
        "descripcion": {
            "use": "Universal Sentence Encoder",
            "st1": "all-mpnet-base-v2",
            "st2": "all-MiniLM-L6-v2", 
            "st3": "paraphrase-mpnet-base-v2"
        }
    }

@app.post("/cargar-modelo/{nombre_modelo}")
async def cargar_modelo_endpoint(nombre_modelo: str):
    """Carga un modelo específico"""
    try:
        info_modelo = cargar_modelo(nombre_modelo)
        return {
            "mensaje": f"Modelo {nombre_modelo} cargado exitosamente",
            "info": info_modelo
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generar-embeddings", response_model=EmbeddingResponse)
async def generar_embeddings(request: EmbeddingRequest):
    """Genera embeddings para una lista de textos"""
    try:
        # Cargar modelo si no está cargado
        cargar_modelo(request.modelo)
        
        # Generar embeddings
        start_time = time.time()
        nombre, _, _ = MODELOS_DISPONIBLES[request.modelo]
        
        embeddings = manager.embed(
            nombre, 
            request.textos, 
            save=request.guardar, 
            save_csv=False
        )
        
        tiempo_procesamiento = time.time() - start_time
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimensiones=embeddings.shape[1],
            modelo_usado=request.modelo,
            tiempo_procesamiento=tiempo_procesamiento
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generar-dataset")
async def generar_dataset(request: GenerarDatasetRequest, background_tasks: BackgroundTasks):
    """Genera embeddings para todo el dataset (proceso en background)"""
    try:
        # Validar que existe el archivo
        if not os.path.exists(request.archivo_csv):
            raise HTTPException(status_code=404, detail="Archivo CSV no encontrado")
        
        # Agregar tarea en background
        background_tasks.add_task(
            procesar_dataset_background, 
            request.archivo_csv,
            request.modelo,
            request.columnas,
            request.incluir_reference_observation
        )
        
        return {
            "mensaje": "Procesamiento de dataset iniciado en background",
            "modelo": request.modelo,
            "archivo": request.archivo_csv,
            "estado": "en_proceso"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def procesar_dataset_background(
    archivo_csv: str, 
    modelo: str, 
    columnas: List[str],
    incluir_reference_observation: bool
):
    """Procesa el dataset completo en background"""
    try:
        print(f"Iniciando procesamiento de dataset con modelo {modelo}")
        
        # Cargar datos
        start_time = time.time()
        data_sample = pd.read_csv(archivo_csv)
        tiempo_carga = time.time() - start_time
        print(f"Dataset cargado en {tiempo_carga:.2f} segundos")
        
        # Preparar columnas adicionales si se solicita
        reference_column = None
        observation_column = None
        if incluir_reference_observation and 'Reference' in data_sample.columns and 'Observation' in data_sample.columns:
            reference_column = data_sample['Reference'].copy()
            observation_column = data_sample['Observation'].copy()
        
        # Procesar texto
        start_time = time.time()
        columnas_embedding = data_sample[columnas]
        all_intents = columnas_embedding.astype(str).map(lambda x: x.replace(' ', '_')).agg(' '.join, axis=1).tolist()
        tiempo_procesado = time.time() - start_time
        print(f"Texto procesado en {tiempo_procesado:.2f} segundos")
        
        # Cargar modelo
        cargar_modelo(modelo)
        
        # Generar embeddings
        start_time = time.time()
        nombre, _, _ = MODELOS_DISPONIBLES[modelo]
        embeddings = manager.embed(nombre, all_intents, save=True, save_csv=False)
        tiempo_embeddings = time.time() - start_time
        print(f"Embeddings generados en {tiempo_embeddings:.2f} segundos")
        
        # Crear DataFrame final
        embeddings_df = pd.DataFrame(embeddings)
        
        if reference_column is not None and observation_column is not None:
            final_df = pd.DataFrame({
                'Reference': reference_column,
                'Observation': observation_column
            })
            final_df = pd.concat([final_df, embeddings_df], axis=1)
        else:
            final_df = embeddings_df
        
        # Guardar resultado
        model_dir = os.path.join(f"{path_test}/embeddings", nombre)
        os.makedirs(model_dir, exist_ok=True)
        save_csv_path = os.path.join(model_dir, f"{nombre}_complete.csv")
        final_df.to_csv(save_csv_path, index=False)
        
        # Guardar tiempos
        times = pd.DataFrame([{
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "modelo": nombre,
            "tiempo_carga_dataset": round(tiempo_carga, 2),
            "tiempo_procesado_texto": round(tiempo_procesado, 2),
            "tiempo_generacion_embeddings": round(tiempo_embeddings, 2),
            "via_api": True
        }])
        
        csv_path = f"{path_test}/embeddings/times_embeddings.csv"
        os.makedirs(f"{path_test}/embeddings", exist_ok=True)
        
        if os.path.exists(csv_path):
            times_ant = pd.read_csv(csv_path)
            times = pd.concat([times_ant, times], ignore_index=True)
        
        times.to_csv(csv_path, index=False)
        
        print(f"Procesamiento completado. Archivo guardado en: {save_csv_path}")
        
    except Exception as e:
        print(f"Error en procesamiento background: {str(e)}")

@app.get("/estado-procesamiento")
async def estado_procesamiento():
    """Obtiene el estado del procesamiento actual"""
    return {
        "modelos_cargados": list(modelos_cargados.keys()),
        "timestamp": datetime.now().isoformat(),
        "directorio_salida": f"{path_test}/embeddings"
    }

@app.get("/descargar-embeddings/{modelo}")
async def descargar_embeddings(modelo: str):
    """Descarga el archivo de embeddings generado"""
    if modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=400, detail="Modelo no válido")
    
    nombre, _, _ = MODELOS_DISPONIBLES[modelo]
    archivo_path = f"{path_test}/embeddings/{nombre}/{nombre}_complete.csv"
    
    if not os.path.exists(archivo_path):
        raise HTTPException(status_code=404, detail="Archivo de embeddings no encontrado")
    
    return FileResponse(
        archivo_path, 
        filename=f"{nombre}_embeddings.csv",
        media_type="text/csv"
    )

@app.delete("/limpiar-modelos")
async def limpiar_modelos():
    """Limpia todos los modelos cargados de la memoria"""
    global modelos_cargados, manager
    modelos_anteriores = list(modelos_cargados.keys())
    modelos_cargados.clear()
    manager = None
    
    return {
        "mensaje": "Modelos limpiados de la memoria",
        "modelos_anteriores": modelos_anteriores
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
# JCMG Embedding Generator API

API REST construida con FastAPI para generar embeddings usando diferentes modelos de transformers.

## 🚀 Inicio Rápido

### Con Docker (Recomendado)

```bash
# Construir y ejecutar la API
docker-compose up --build

# La API estará disponible en: http://localhost:8000
```

### Instalación Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la API
cd src/EmbeddingGenerator
python main.py
```

## 📊 Modelos Disponibles

- **use**: Universal Sentence Encoder
- **st1**: all-mpnet-base-v2  
- **st2**: all-MiniLM-L6-v2
- **st3**: paraphrase-mpnet-base-v2

## 🔌 Endpoints de la API

### GET `/`
Estado general de la API

### GET `/modelos`
Lista todos los modelos disponibles y cargados

### POST `/cargar-modelo/{nombre_modelo}`
Carga un modelo específico en memoria

### POST `/generar-embeddings`
Genera embeddings para textos específicos

**Ejemplo de request:**
```json
{
  "textos": ["Mexico.Total 2012 Mujeres.>65", "USA.California 2020 Hombres.25-30"],
  "modelo": "st1",
  "guardar": false
}
```

### POST `/generar-dataset`
Procesa un dataset completo (en background)

**Ejemplo de request:**
```json
{
  "archivo_csv": "../data/sample.csv",
  "modelo": "st1",
  "columnas": ["Spatial", "Temporal", "Interest"],
  "incluir_reference_observation": true
}
```

### GET `/descargar-embeddings/{modelo}`
Descarga el archivo CSV con embeddings generados

### DELETE `/limpiar-modelos`
Limpia todos los modelos de la memoria

## 🐳 Uso con Docker

### API (Recomendado)
```bash
# Ejecutar API
docker-compose up

# Acceder a la documentación interactiva
# http://localhost:8000/docs
```

### CLI (Compatibilidad)
```bash
# Ejecutar generación por CLI
docker-compose --profile cli up embedding-st1

# Ejecutar todos los modelos
docker-compose --profile cli up embedding-use embedding-st2 embedding-st3
```

## 📝 Ejemplos de Uso

### Generar embeddings simples
```python
import requests

# Cargar modelo
response = requests.post("http://localhost:8000/cargar-modelo/st1")

# Generar embeddings
data = {
    "textos": ["Mexico.Total 2012 Mujeres.>65"],
    "modelo": "st1"
}
response = requests.post("http://localhost:8000/generar-embeddings", json=data)
embeddings = response.json()["embeddings"]
```

### Procesar dataset completo
```python
import requests

data = {
    "archivo_csv": "../data/sample.csv",
    "modelo": "st1",
    "columnas": ["Spatial", "Temporal", "Interest"]
}
response = requests.post("http://localhost:8000/generar-dataset", json=data)
```

## 🔧 Configuración

### Variables de Entorno
- `PYTHONUNBUFFERED=1`: Para logs en tiempo real

### Volúmenes Docker
- `./test:/app/test`: Directorio de salida
- `./data:/app/data:ro`: Datos de entrada (solo lectura)

## 📚 Documentación API

Una vez ejecutando, visita:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Desarrollo

### Estructura del Proyecto
```
src/
├── EmbeddingGenerator/
│   └── main.py              # FastAPI app
├── Modules/
│   └── model_manager.py     # Gestión de modelos
├── generate_embedding.py    # Script CLI original
└── ...
```

### Ejecutar en Desarrollo
```bash
cd src/EmbeddingGenerator
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔍 Monitoreo

### Logs
```bash
# Ver logs de la API
docker-compose logs -f jcmg-embedding-api

# Ver logs de procesamiento CLI
docker-compose --profile cli logs -f
```

### Estado
```bash
# Estado de la API
curl http://localhost:8000/

# Modelos cargados
curl http://localhost:8000/modelos
```

## 📊 Archivos de Salida

Los embeddings se guardan en:
- `test/embeddings/{modelo}/{modelo}_complete.csv`
- `test/embeddings/times_embeddings.csv` (tiempos de procesamiento)

## ⚡ Performance Tips

1. **Precarga modelos**: Usa `/cargar-modelo/{modelo}` antes de generar embeddings
2. **Procesamiento en batch**: Envía múltiples textos en una sola petición
3. **Background tasks**: Usa `/generar-dataset` para datasets grandes
4. **Limpieza de memoria**: Usa `/limpiar-modelos` cuando cambies de modelo

## 🚨 Troubleshooting

### Problemas Comunes

**Puerto ocupado:**
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8001:8000"  # Usar puerto 8001 externamente
```

**Memoria insuficiente:**
```bash
# Limpiar modelos cargados
curl -X DELETE http://localhost:8000/limpiar-modelos
```

**Archivos no encontrados:**
- Verificar que `data/sample.csv` existe
- Verificar permisos de escritura en `test/`

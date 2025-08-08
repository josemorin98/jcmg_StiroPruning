# JCMG Embedding & Clustering System

Sistema completo para generar embeddings, realizar clustering y predicción de grupos usando diferentes modelos de transformers.

## 🚀 Características Principales

- **Generación de Embeddings**: Soporte para múltiples modelos (USE, Sentence Transformers)
- **Clustering Inteligente**: UMAP + HDBSCAN con optimización de hiperparámetros
- **Predicción por Similitud**: Búsqueda directa en embeddings originales
- **Clasificación**: Modelos de ML para predicción de grupos
- **Métricas DBCV**: Validación de calidad de clustering
- **Trazabilidad**: Mapeo de resultados a vectores originales

## 📊 Modelos Disponibles

- **use**: Universal Sentence Encoder
- **st1**: all-mpnet-base-v2  
- **st2**: all-MiniLM-L6-v2
- **st3**: paraphrase-mpnet-base-v2

## � Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd jcmg_StiroPruning

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Uso Principal

### 1. Generar Embeddings

```bash
# Generar embeddings con modelo específico
cd src
python generate_embedding.py --modelo st1

# Generar con diferentes modelos
python generate_embedding.py --modelo use
python generate_embedding.py --modelo st2
python generate_embedding.py --modelo st3
```

### 2. Realizar Clustering

```bash
# Búsqueda de hiperparámetros con diferentes métodos
cd src
python find_hyperparams.py --modelo st1 --method bayesian
python find_hyperparams.py --modelo st1 --method random
python find_hyperparams.py --modelo st1 --method separate_grid
```

### 3. Predicción de Grupos

```bash
# Predicción usando clasificadores ML
python predict.py --modelo st1 --params bayesian

# Con embeddings ajustados
python predict.py --modelo st1 --params bayesian --use_adjusted
```

## 🔍 Funcionalidades Avanzadas

### **Predicción por Similitud Directa**
- Búsqueda en embeddings originales sin transformaciones UMAP
- Comparación directa con similitud coseno
- Trazabilidad completa hasta vectores originales del CSV

### **Optimización de Hiperparámetros**
- **Random Search**: Exploración aleatoria del espacio de parámetros
- **Bayesian Search**: Optimización inteligente con Hyperopt
- **Grid Search**: Búsqueda exhaustiva en grilla

### **Métricas de Calidad**
- **DBCV**: Density-Based Cluster Validation
- **Silhouette Score**: Cohesión y separación de clusters
- **Estadísticas de Clustering**: Distribución y características

### **Múltiples Enfoques de Predicción**
1. **Clasificadores ML**: Random Forest, MLP, XGBoost
2. **Similitud Directa**: Búsqueda en espacio original
3. **Búsqueda por Grupo**: Similares dentro de clusters específicos

## � Estructura del Proyecto

```
src/
├── generate_embedding.py       # Generación de embeddings
├── find_hyperparams.py        # Optimización de clustering
├── predict.py                 # Predicción de grupos
├── Modules/
│   ├── model_manager.py       # Gestión de modelos de embeddings
│   ├── clustering_manager.py  # Clustering UMAP + HDBSCAN
│   ├── classification_manager.py # Clasificadores ML
│   ├── predict_vector.py      # Predicción por similitud
│   ├── estimators.py          # Estimadores personalizados
│   └── grid_search.py         # Búsqueda en grilla
data/
├── sample.csv                 # Dataset de entrada
test/
├── embeddings/               # Embeddings generados
├── Modelos_{modelo}/        # Modelos y resultados de clustering
│   ├── embeddings_originales_labeled_{method}.csv
│   ├── embeddings_umap_labeled_{method}.csv
│   ├── umap_{method}.pkl
│   ├── hdbscan_{method}.pkl
│   └── dbcv_results_{method}.csv
```

## 🔄 Flujo de Trabajo Completo

### 1. **Preparación de Datos**
```bash
# El CSV debe contener columnas: Spatial, Temporal, Interest, Reference, Observation
# Ejemplo de fila: Mexico.Total, 2012, Mujeres.>65, 0.139, 0.139
```

### 2. **Generación de Embeddings**
```bash
python generate_embedding.py --modelo st1
# Genera: test/embeddings/st1/st1_complete.csv
```

### 3. **Clustering y Optimización**
```bash
python find_hyperparams.py --modelo st1 --method bayesian
# Genera: test/Modelos_st1/embeddings_originales_labeled_bayesian.csv
#         test/Modelos_st1/umap_bayesian.pkl
#         test/Modelos_st1/hdbscan_bayesian.pkl
#         test/Modelos_st1/dbcv_results_bayesian.csv
```

### 4. **Predicción y Análisis**
```bash
python predict.py --modelo st1 --params bayesian
# Muestra:
# - Grupo predicho
# - Similares en el grupo
# - Vectores originales correspondientes
# - Métricas de confianza
```

## � Métricas y Resultados

### **Archivos de Salida:**

1. **Embeddings Etiquetados**: `embeddings_originales_labeled_{method}.csv`
   - Embeddings + etiquetas de clustering
   - Base para predicción por similitud

2. **Modelos Entrenados**: `umap_{method}.pkl`, `hdbscan_{method}.pkl`
   - Modelos UMAP y HDBSCAN optimizados
   - Reproducibilidad de resultados

3. **Métricas DBCV**: `dbcv_results_{method}.csv`
   - Validación de calidad de clustering
   - Comparación entre métodos

4. **Tiempos de Ejecución**: `times_*.csv`
   - Rendimiento por método
   - Optimización de recursos

## 🎯 Casos de Uso

### **Predicción de Nuevos Vectores**
```python
# Vector ejemplo
vector_input = ["Mexico.Total", "2013", "Mujeres.>65", "0.139", "0.139"]

# El sistema automáticamente:
# 1. Genera embedding del vector
# 2. Encuentra grupo más similar
# 3. Muestra vectores originales similares
# 4. Proporciona métricas de confianza
```

### **Análisis de Similitud**
- Búsqueda exacta en grupos específicos
- Top-N más similares dentro de clusters
- Trazabilidad hasta datos originales
- Métricas de distancia y similitud

### **Validación de Clustering**
- Métricas DBCV por método
- Comparación de hiperparámetros
- Análisis de distribución de clusters
- Detección de outliers/ruido

## 🔧 Configuración Avanzada

### **Embeddings Ajustados**
```bash
# Usar solo columnas Spatial, Temporal, Interest
python generate_embedding.py --modelo st1
python predict.py --modelo st1 --use_adjusted
```

### **Diferentes Métodos de Clustering**
```bash
# Bayesian optimization (recomendado)
python find_hyperparams.py --method bayesian

# Random search (rápido)
python find_hyperparams.py --method random

# Grid search (exhaustivo)
python find_hyperparams.py --method separate_grid
```

## 🚨 Troubleshooting

### **Problemas Comunes**

**Error de dimensionalidad:**
- Verificar que el vector de entrada coincida con el modo (ajustado/normal)
- 5 elementos para modo normal, 3 para ajustado

**Archivos no encontrados:**
- Verificar que se hayan generado embeddings primero
- Verificar que exista el CSV original en `data/sample.csv`

**Memoria insuficiente:**
- Usar modelos más pequeños (st2 en lugar de st1)
- Reducir tamaño del dataset
- Usar search methods más eficientes

## 📊 Rendimiento

- **st2 (MiniLM)**: Más rápido, menor precisión
- **st1 (MPNet)**: Balance óptimo velocidad/precisión
- **use**: Buena precisión, requiere más memoria
- **st3 (Paraphrase)**: Mejor para similitud semántica

## 🔄 Actualizaciones Recientes

- ✅ Predicción directa por similitud (sin UMAP)
- ✅ Trazabilidad a vectores originales del CSV
- ✅ Métricas DBCV integradas
- ✅ Soporte para embeddings ajustados
- ✅ Clasificadores ML múltiples
- ✅ Búsqueda optimizada de hiperparámetros

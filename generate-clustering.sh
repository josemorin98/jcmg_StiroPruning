#!/bin/bash

# Verificar que se proporcionen los parámetros requeridos
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "Uso: $0 <max_evals> <label_lower> <label_upper> [use_adjusted]"
    echo "Parámetros:"
    echo "  max_evals: Número de evaluaciones"
    echo "  label_lower: Límite inferior para las etiquetas"
    echo "  label_upper: Límite superior para las etiquetas"
    echo "  use_adjusted: 'true' para usar embeddings ajustados, 'false' o vacío para originales"
    echo ""
    echo "Ejemplos:"
    echo "  $0 100 500 1000           # Usar embeddings originales"
    echo "  $0 100 500 1000 true      # Usar embeddings ajustados"
    echo "  $0 100 500 1000 false     # Usar embeddings originales"
    exit 1
fi

# Asignar parámetros a variables
MAX_EVALS=$1
LABEL_LOWER=$2
LABEL_UPPER=$3
USE_ADJUSTED=${4:-false}  # Por defecto false si no se proporciona

# Determinar si usar el flag --use_adjusted
if [ "$USE_ADJUSTED" = "true" ]; then
    ADJUSTED_FLAG="--use_adjusted"
    ADJUSTED_TEXT="(con ajuste spatial/temporal/interest)"
else
    ADJUSTED_FLAG=""
    ADJUSTED_TEXT="(embeddings originales)"
fi

echo "Ejecutando clustering con parámetros:"
echo "  max_evals: $MAX_EVALS"
echo "  label_lower: $LABEL_LOWER"
echo "  label_upper: $LABEL_UPPER"
echo "  embeddings ajustados: $USE_ADJUSTED $ADJUSTED_TEXT"
echo ""

cd src

echo "=== Iniciando búsqueda de hiperparámetros ==="

# Ejecutar para todos los modelos con los parámetros especificados
# python find_hyperparams.py --modelo use --max_evals $MAX_EVALS --label_lower $LABEL_LOWER --label_upper $LABEL_UPPER $ADJUSTED_FLAG
python find_hyperparams.py --modelo st1 --max_evals $MAX_EVALS --label_lower $LABEL_LOWER --label_upper $LABEL_UPPER $ADJUSTED_FLAG
python find_hyperparams.py --modelo st2 --max_evals $MAX_EVALS --label_lower $LABEL_LOWER --label_upper $LABEL_UPPER $ADJUSTED_FLAG
python find_hyperparams.py --modelo st3 --max_evals $MAX_EVALS --label_lower $LABEL_LOWER --label_upper $LABEL_UPPER $ADJUSTED_FLAG

echo ""
echo "=== Clustering completado ==="

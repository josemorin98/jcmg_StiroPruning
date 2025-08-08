#!/bin/bash

# Script de ayuda para ejecutar el generador de embeddings con Docker

echo "=== JCMG Embedding Generator - Docker Helper ==="
echo ""

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [COMANDO] [OPCIONES]"
    echo ""
    echo "Comandos disponibles:"
    echo "  build         Construir la imagen Docker"
    echo "  run [modelo]  Ejecutar generación de embeddings con modelo específico"
    echo "  all           Ejecutar generación para todos los modelos"
    echo "  clean         Limpiar contenedores e imágenes"
    echo "  logs          Mostrar logs del contenedor"
    echo "  shell         Abrir shell interactivo en el contenedor"
    echo ""
    echo "Modelos disponibles: use, st1, st2, st3"
    echo ""
    echo "Ejemplos:"
    echo "  $0 build"
    echo "  $0 run st1"
    echo "  $0 run use"
    echo "  $0 all"
    echo ""
}

# Función para construir la imagen
build_image() {
    echo "🔨 Construyendo imagen Docker..."
    docker build -t jcmg-embedding-generator .
    echo "✅ Imagen construida exitosamente"
}

# Función para ejecutar con un modelo específico
run_model() {
    local modelo=$1
    if [[ ! "$modelo" =~ ^(use|st1|st2|st3)$ ]]; then
        echo "❌ Modelo no válido: $modelo"
        echo "Modelos disponibles: use, st1, st2, st3"
        exit 1
    fi
    
    echo "🚀 Ejecutando generación de embeddings con modelo: $modelo"
    docker run --rm \
        -v "$(pwd)/test:/app/test" \
        -v "$(pwd)/data:/app/data:ro" \
        -e PYTHONUNBUFFERED=1 \
        --name "jcmg-embedding-$modelo" \
        jcmg-embedding-generator \
        python generate_embedding.py --modelo "$modelo"
    echo "✅ Generación completada para modelo: $modelo"
}

# Función para ejecutar todos los modelos
run_all() {
    echo "🚀 Ejecutando generación para todos los modelos..."
    for modelo in use st1 st2 st3; do
        echo ""
        echo "📊 Procesando modelo: $modelo"
        run_model "$modelo"
        echo "✅ Completado: $modelo"
        echo "─────────────────────────────────────────"
    done
    echo "🎉 Todos los modelos completados!"
}

# Función para limpiar
clean() {
    echo "🧹 Limpiando contenedores e imágenes..."
    docker container prune -f
    docker rmi jcmg-embedding-generator 2>/dev/null || true
    echo "✅ Limpieza completada"
}

# Función para mostrar logs
show_logs() {
    echo "📋 Mostrando logs..."
    docker logs jcmg-embedding-gen 2>/dev/null || echo "No hay contenedor activo"
}

# Función para shell interactivo
interactive_shell() {
    echo "🐚 Abriendo shell interactivo..."
    docker run --rm -it \
        -v "$(pwd)/test:/app/test" \
        -v "$(pwd)/data:/app/data:ro" \
        jcmg-embedding-generator \
        /bin/bash
}

# Procesamiento de argumentos
case $1 in
    build)
        build_image
        ;;
    run)
        if [ -z "$2" ]; then
            echo "❌ Especifica un modelo: use, st1, st2, st3"
            exit 1
        fi
        run_model "$2"
        ;;
    all)
        run_all
        ;;
    clean)
        clean
        ;;
    logs)
        show_logs
        ;;
    shell)
        interactive_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Comando no reconocido: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

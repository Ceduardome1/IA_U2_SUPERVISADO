#!/usr/bin/env bash
# Script conveniente para ejecutar el proyecto usando el entorno virtual con Python 3.14
VENV_PY="/home/cesar/.virtualenvs/ia_supervisado/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "Error: No se encontró el entorno virtual en $VENV_PY"
    exit 1
fi

exec "$VENV_PY" "$@"

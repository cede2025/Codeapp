#!/bin/bash

# AI Command Center - Production Dockerfile Generation Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje tworzenie zoptymalizowanych, produkcyjnych
#       plików Dockerfile dla serwera FastAPI i workera Celery, wraz z
#       odpowiednim plikiem .dockerignore.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_PATH="backend"
readonly DOCKERFILE_FASTAPI="${BACKEND_PATH}/Dockerfile.fastapi"
readonly DOCKERFILE_CELERY="${BACKEND_PATH}/Dockerfile.celery"
readonly DOCKERIGNORE_FILE="${BACKEND_PATH}/.dockerignore"

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# --- Definicje Kodu do Wstrzyknięcia ---

read -r -d '' DOCKERFILE_FASTAPI_CONTENT <<'EOF'
# Dockerfile for FastAPI Application (Production Ready)
# Etap 1: Budowanie środowiska z zależnościami
FROM python:3.11-slim as builder

# Ustawienia środowiskowe dla Pythona
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja zależności systemowych, jeśli są potrzebne (np. dla psycopg2)
# RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Kopiowanie tylko pliku z zależnościami w celu wykorzystania cache'u Docker
COPY requirements.txt .

# Instalacja zależności
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Etap 2: Budowanie finalnego, lekkiego obrazu
FROM python:3.11-slim

# Ustawienia środowiskowe
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Kopiowanie pre-kompilowanych zależności z etapu buildera
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Tworzenie użytkownika bez uprawnień roota dla bezpieczeństwa
RUN addgroup --system app && adduser --system --group app
USER app

# Kopiowanie kodu aplikacji
# Upewnij się, że .dockerignore jest poprawnie skonfigurowany
COPY . .

# Ekspozycja portu, na którym działa Uvicorn
EXPOSE 8000

# Komenda startowa dla serwera FastAPI
# Używamy gunicorn jako process managera dla uvicorn dla większej stabilności w produkcji
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn_conf.py", "app.main:app"]
EOF

read -r -d '' DOCKERFILE_CELERY_CONTENT <<'EOF'
# Dockerfile for Celery Worker (Production Ready)
# Ten plik jest bardzo podobny do Dockerfile.fastapi, aby współdzielić warstwy cache.
# Różni się głównie komendą startową (CMD).

# Etap 1: Budowanie środowiska z zależnościami (identyczny jak dla FastAPI)
FROM python:3.11-slim as builder
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Etap 2: Budowanie finalnego, lekkiego obrazu
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Tworzenie użytkownika bez uprawnień roota
RUN addgroup --system app && adduser --system --group app
USER app

# Kopiowanie kodu aplikacji
COPY . .

# Komenda startowa dla workera Celery
# Używamy -P gevent dla asynchronicznych zadań I/O-bound
CMD ["celery", "-A", "app.tasks.celery_app", "worker", "-l", "info", "-P", "gevent"]
EOF

read -r -d '' DOCKERIGNORE_CONTENT <<'EOF'
# Pliki i katalogi specyficzne dla Gita
.git
.gitignore

# Pliki i katalogi specyficzne dla Pythona
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
*.so

# Środowiska wirtualne
.venv/
venv/
env/

# Pliki IDE
.vscode/
.idea/
*.swp

# Pliki systemowe
.DS_Store
Thumbs.db

# Pliki konfiguracyjne i logi lokalne
.env
*.log
*.log.*
celerybeat-schedule

# Pliki testowe i raporty pokrycia
.pytest_cache/
htmlcov/
.coverage
*.cover

# Skrypty i pliki tymczasowe
scripts/
*.bak
*.tmp
EOF

read -r -d '' GUNICORN_CONF_CONTENT <<'EOF'
# backend/gunicorn_conf.py

import os

# Ustawienia Gunicorn
bind = "0.0.0.0:8000"
workers = int(os.environ.get('GUNICORN_WORKERS', '2'))
worker_class = "uvicorn.workers.UvicornWorker"
threads = int(os.environ.get('GUNICORN_THREADS', '4'))

# Logowanie
accesslog = "-"  # Logi dostępu do stdout
errorlog = "-"   # Logi błędów do stdout
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')

# Inne ustawienia
# gracefull_timeout, timeout, keepalive, etc.
EOF

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Production Dockerfile Generation"
    echo "-----------------------------------------------------------------"
    
    # Krok 1: Weryfikacja
    if [ ! -d "${BACKEND_PATH}" ]; then
        error_message "Katalog '${BACKEND_PATH}' nie został znaleziony. Uruchom ten skrypt z głównego katalogu projektu."
        exit 1
    fi

    if ! confirm "This script will create/overwrite Dockerfiles in '${BACKEND_PATH}'. Continue?"; then
        log_message "Operation cancelled by user."
        exit 0
    fi

    # Krok 2: Tworzenie/nadpisywanie plików
    log_message "Generating Dockerfiles and supporting files..."

    # Kopia zapasowa i tworzenie Dockerfile.fastapi
    if [ -f "$DOCKERFILE_FASTAPI" ]; then mv "$DOCKERFILE_FASTAPI" "$DOCKERFILE_FASTAPI.bak"; fi
    echo "$DOCKERFILE_FASTAPI_CONTENT" > "$DOCKERFILE_FASTAPI"
    success_message "Created '${DOCKERFILE_FASTAPI}' (previous version saved as .bak)."

    # Kopia zapasowa i tworzenie Dockerfile.celery
    if [ -f "$DOCKERFILE_CELERY" ]; then mv "$DOCKERFILE_CELERY" "$DOCKERFILE_CELERY.bak"; fi
    echo "$DOCKERFILE_CELERY_CONTENT" > "$DOCKERFILE_CELERY"
    success_message "Created '${DOCKERFILE_CELERY}' (previous version saved as .bak)."

    # Tworzenie .dockerignore
    echo "$DOCKERIGNORE_CONTENT" > "$DOCKERIGNORE_FILE"
    success_message "Created/Updated '${DOCKERIGNORE_FILE}'."
    
    # Tworzenie pliku konfiguracyjnego Gunicorn
    echo "$GUNICORN_CONF_CONTENT" > "${BACKEND_PATH}/gunicorn_conf.py"
    success_message "Created '${BACKEND_PATH}/gunicorn_conf.py'."

    # Krok 3: Instrukcje końcowe
    echo
    warning_message "------------------------- ACTION REQUIRED -------------------------"
    warning_message "Dockerfiles have been generated. Here's how to use them:"
    echo
    echo -e "  \033[1;33m1. To build the FastAPI server image:\033[0m"
    echo -e "     \033[1;32mdocker build -t aicc-fastapi-server:latest -f ${DOCKERFILE_FASTAPI} ./${BACKEND_PATH}\033[0m"
    echo
    echo -e "  \033[1;33m2. To build the Celery worker image:\033[0m"
    echo -e "     \033[1;32mdocker build -t aicc-celery-worker:latest -f ${DOCKERFILE_CELERY} ./${BACKEND_PATH}\033[0m"
    echo
    warning_message "Next Step: Create a 'docker-compose.yml' file to orchestrate running"
    warning_message "these images along with your PostgreSQL and Redis containers."
    warning_message "-----------------------------------------------------------------"
    echo
    success_message "Dockerfile generation complete!"
}

# Uruchomienie funkcji głównej
main
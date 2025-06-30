#!/bin/bash

# AI Command Center - Development Docker Compose Generation Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje tworzenie pliku docker-compose.yml
#       do uruchomienia całego środowiska deweloperskiego (FastAPI, Celery,
#       PostgreSQL, Redis) w kontenerach Docker.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly COMPOSE_FILE="docker-compose.yml"
readonly BACKEND_PATH="backend"
readonly ENV_FILE="${BACKEND_PATH}/.env"
readonly DOCKERFILE_FASTAPI="${BACKEND_PATH}/Dockerfile.fastapi"
readonly DOCKERFILE_CELERY="${BACKEND_PATH}/Dockerfile.celery"

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# Funkcja do odczytywania wartości z pliku .env
get_env_var() {
    local var_name="$1"
    local env_file="$2"
    # Użyj grep i cut, aby wyodrębnić wartość po znaku '='
    grep -E "^${var_name}=" "${env_file}" | cut -d'=' -f2 | sed 's/"//g'
}

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Docker Compose Generation"
    echo "-----------------------------------------------------------------"
    
    # Krok 1: Weryfikacja
    log_message "Verifying required files..."
    for file in "$ENV_FILE" "$DOCKERFILE_FASTAPI" "$DOCKERFILE_CELERY"; do
        if [ ! -f "$file" ]; then
            error_message "Required file not found: $file. Please run previous scripts to generate it."
            exit 1
        fi
    done
    success_message "All required files are present."

    if [ -f "$COMPOSE_FILE" ]; then
        if ! confirm "File '${COMPOSE_FILE}' already exists. Overwrite it?"; then
            log_message "Operation cancelled by user."
            exit 0
        fi
        mv "$COMPOSE_FILE" "$COMPOSE_FILE.bak"
        log_message "Backed up existing '${COMPOSE_FILE}' to '${COMPOSE_FILE}.bak'."
    fi

    # Krok 2: Odczytanie konfiguracji z .env
    log_message "Reading database configuration from '${ENV_FILE}'..."
    DB_PORT=$(get_env_var "DB_PORT" "$ENV_FILE")
    if [ -z "$DB_PORT" ]; then
        error_message "DB_PORT not found in '${ENV_FILE}'. Please ensure it is configured."
        exit 1
    fi
    log_message "Database will be exposed on host port: ${DB_PORT}"

    # Krok 3: Generowanie pliku docker-compose.yml
    log_message "Generating '${COMPOSE_FILE}'..."

    cat << EOF > "${COMPOSE_FILE}"
# Docker Compose for AI Command Center (Development Environment)
# This file orchestrates all necessary services for local development.
# To run: docker-compose up --build
# To stop: docker-compose down

version: '3.8'

services:
  # --- Baza Danych PostgreSQL ---
  postgres_db:
    image: postgres:15-alpine
    container_name: aicc-postgres-db
    # Zmienne środowiskowe są pobierane z pliku .env w katalogu backend
    env_file:
      - ${BACKEND_PATH}/.env
    ports:
      # Mapuje port kontenera (5432) na port hosta zdefiniowany w .env
      - "${DB_PORT}:5432"
    volumes:
      # Wolumin nazwany do trwałego przechowywania danych bazy
      - aicc_postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U \${DB_USER} -d \${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --- Broker Zadań i Cache Redis ---
  redis:
    image: redis:7-alpine
    container_name: aicc-redis
    ports:
      - "6379:6379"
    volumes:
      - aicc_redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --- Serwer API FastAPI ---
  fastapi_server:
    # Buduje obraz z Dockerfile.fastapi
    build:
      context: ./${BACKEND_PATH}
      dockerfile: Dockerfile.fastapi
    container_name: aicc-fastapi-server
    # Przekazuje wszystkie zmienne z pliku .env do kontenera
    env_file:
      - ${BACKEND_PATH}/.env
    # Zmienia hosta w .env na nazwę kontenera, aby aplikacja mogła znaleźć bazę
    environment:
      - DB_HOST=postgres_db
      - REDIS_HOST=redis
    ports:
      - "8000:8000"
    # Zależy od bazy danych i Redis, Docker Compose uruchomi je w odpowiedniej kolejności
    depends_on:
      postgres_db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # --- Worker Celery ---
  celery_worker:
    # Buduje obraz z Dockerfile.celery
    build:
      context: ./${BACKEND_PATH}
      dockerfile: Dockerfile.celery
    container_name: aicc-celery-worker
    env_file:
      - ${BACKEND_PATH}/.env
    environment:
      - DB_HOST=postgres_db
      - REDIS_HOST=redis
    depends_on:
      postgres_db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

# --- Woluminy Nazwane ---
# Definiuje woluminy, aby dane przetrwały restarty kontenerów
volumes:
  aicc_postgres_data:
    driver: local
  aicc_redis_data:
    driver: local
EOF

    success_message "File '${COMPOSE_FILE}' has been generated successfully."

    # Krok 4: Instrukcje końcowe
    echo
    warning_message "------------------------- ACTION REQUIRED -------------------------"
    warning_message "The 'docker-compose.yml' file is ready. Here's how to use it:"
    echo
    echo -e "  \033[1;33m1. To build and start all services in the background:\033[0m"
    echo -e "     \033[1;32mdocker-compose up --build -d\033[0m"
    echo
    echo -e "  \033[1;33m2. To view logs from all running services:\033[0m"
    echo -e "     \033[1;32mdocker-compose logs -f\033[0m"
    echo
    echo -e "  \033[1;33m3. After the first start, apply database migrations:\033[0m"
    echo -e "     \033[1;32mdocker-compose exec fastapi_server alembic upgrade head\033[0m"
    echo
    echo -e "  \033[1;33m4. To stop and remove all services and networks:\033[0m"
    echo -e "     \033[1;32mdocker-compose down\033[0m"
    echo
    warning_message "Your full development environment is now managed by Docker Compose."
    warning_message "-----------------------------------------------------------------"
    echo
}

# Uruchomienie funkcji głównej
main
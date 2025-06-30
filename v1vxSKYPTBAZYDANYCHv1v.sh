#!/bin/bash

# AI Command Center - Database Setup Automation Script
# Wersja: 1.2.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje pełne przygotowanie środowiska bazy danych
#       dla projektu AI Command Center, włączając w to:
#       - Uruchomienie kontenerów Docker (PostgreSQL, Redis) za pomocą docker-compose.
#       - Generowanie bezpiecznych haseł.
#       - Weryfikację połączenia z bazą danych.

# --- Konfiguracja Skryptu ---
set -euo pipefail # Zakończ w przypadku błędu, traktuj niezainicjowane zmienne jako błąd

# --- Zmienne Projektowe ---
readonly COMPOSE_FILE="docker-compose.yml"
readonly ENV_FILE="backend/.env"
readonly CONTAINER_NAME_DB="aicc-postgres-db"
readonly CONTAINER_NAME_REDIS="aicc-redis"

# --- Funkcje Pomocnicze ---
log_message() {
    echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

success_message() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

warning_message() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

error_message() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

confirm() {
    # Użycie -n 1 do odczytania jednego znaku, -r aby nie interpretować backslashy
    read -p "$1 (y/N): " -n 1 -r
    echo # nowa linia
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1 # false
    fi
    return 0 # true
}

check_dependencies() {
    log_message "Sprawdzanie zależności (docker, docker-compose, psql)..."
    local missing_deps=0
    for cmd in docker docker-compose psql; do
        if ! command -v "$cmd" &> /dev/null; then
            error_message "Zależność '$cmd' nie została znaleziona. Proszę ją zainstalować."
            missing_deps=1
        fi
    done
    if [ "$missing_deps" -eq 1 ]; then
        error_message "Proszę zainstalować brakujące zależności i spróbować ponownie."
        exit 1
    fi
    
    if ! docker info > /dev/null 2>&1; then
        error_message "Docker daemon nie jest uruchomiony. Proszę uruchomić Docker i spróbować ponownie."
        exit 1
    fi
    
    log_message "Wszystkie zależności są obecne i Docker jest uruchomiony."
}

# --- Główna Logika Skryptu ---

generate_password() {
    # Generuje silne, losowe hasło
    head /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 24
}

setup_env_file() {
    log_message "Konfiguracja pliku środowiskowego '${ENV_FILE}'..."
    if [ ! -f "${ENV_FILE}" ]; then
        if [ -f "backend/.env.example" ]; then
            cp "backend/.env.example" "${ENV_FILE}"
            log_message "Utworzono plik '${ENV_FILE}' z szablonu '.env.example'."
        else
            error_message "Nie znaleziono pliku 'backend/.env.example'. Nie można utworzyć pliku .env."
            exit 1
        fi
    fi

    # Upewnij się, że zmienne DB są odkomentowane i ustawione
    sed -i'' -e 's/^#\s*\(DB_ENGINE\)/\1/' \
            -e 's/^#\s*\(DB_USER\)/\1/' \
            -e 's/^#\s*\(DB_PASSWORD\)/\1/' \
            -e 's/^#\s*\(DB_HOST\)/\1/' \
            -e 's/^#\s*\(DB_PORT\)/\1/' \
            -e 's/^#\s*\(DB_NAME\)/\1/' \
            -e 's/^#\s*\(DATABASE_URL\)/\1/' "${ENV_FILE}"

    # Ustaw domyślne wartości, jeśli nie są zdefiniowane
    grep -q -E "^DB_USER=" "${ENV_FILE}" || echo "DB_USER=aicc_user" >> "${ENV_FILE}"
    grep -q -E "^DB_NAME=" "${ENV_FILE}" || echo "DB_NAME=aicc_db" >> "${ENV_FILE}"
    grep -q -E "^DB_HOST=" "${ENV_FILE}" || echo "DB_HOST=localhost" >> "${ENV_FILE}"
    grep -q -E "^DB_PORT=" "${ENV_FILE}" || echo "DB_PORT=5432" >> "${ENV_FILE}"

    # Ustaw hasło, jeśli nie istnieje
    if ! grep -q -E "^DB_PASSWORD=" "${ENV_FILE}"; then
        local new_password
        new_password=$(generate_password)
        echo "DB_PASSWORD=${new_password}" >> "${ENV_FILE}"
        log_message "Wygenerowano i zapisano nowe, bezpieczne hasło do bazy danych."
    fi
    
    # Upewnij się, że DATABASE_URL jest poprawnie skonstruowane
    sed -i'' -e 's|^DATABASE_URL=.*|DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"|' "${ENV_FILE}"

    log_message "Plik .env został skonfigurowany."
}

create_compose_file() {
    log_message "Tworzenie pliku docker-compose.yml..."
    
    # Załaduj zmienne z .env, aby użyć ich w pliku compose
    # shellcheck source=backend/.env
    source "${ENV_FILE}"

    cat << EOF > "${COMPOSE_FILE}"
version: '3.8'

services:
  postgres_db:
    image: postgres:15-alpine
    container_name: ${CONTAINER_NAME_DB}
    environment:
      POSTGRES_USER: \${DB_USER}
      POSTGRES_PASSWORD: \${DB_PASSWORD}
      POSTGRES_DB: \${DB_NAME}
    ports:
      - "\${DB_PORT}:5432"
    volumes:
      - aicc_postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U \${POSTGRES_USER} -d \${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: ${CONTAINER_NAME_REDIS}
    ports:
      - "6379:6379"
    volumes:
      - aicc_redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  aicc_postgres_data:
  aicc_redis_data:
EOF
    success_message "Plik '${COMPOSE_FILE}' został utworzony."
}

start_containers() {
    log_message "Uruchamianie kontenerów Docker (PostgreSQL i Redis)..."
    if [ "$(docker ps -q -f name=^/${CONTAINER_NAME_DB}$)" ]; then
        warning_message "Kontener bazy danych '${CONTAINER_NAME_DB}' już działa."
        if confirm "Czy chcesz go zatrzymać i uruchomić ponownie (dane zostaną zachowane)?"; then
            docker-compose down
        else
            log_message "Pominięto ponowne uruchomienie kontenerów."
            return
        fi
    fi
    
    docker-compose up -d
    
    log_message "Oczekiwanie na pełne uruchomienie i gotowość bazy danych..."
    local retries=20
    local count=0
    while [ $count -lt $retries ]; do
        if docker-compose ps | grep "${CONTAINER_NAME_DB}" | grep -q "healthy"; then
            success_message "Kontener bazy danych '${CONTAINER_NAME_DB}' jest uruchomiony i zdrowy."
            return
        fi
        ((count++))
        echo -n "."
        sleep 2
    done
    
    error_message "Baza danych nie osiągnęła statusu 'healthy' w wyznaczonym czasie."
    docker-compose logs postgres_db
    exit 1
}

verify_db_connection() {
    log_message "Weryfikacja połączenia z bazą danych za pomocą psql..."
    
    # shellcheck source=backend/.env
    source "${ENV_FILE}"
    
    export PGPASSWORD="${DB_PASSWORD}"
    
    if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "\l" > /dev/null; then
        success_message "Połączenie z bazą danych '${DB_NAME}' zostało pomyślnie nawiązane."
    else
        error_message "Nie udało się połączyć z bazą danych za pomocą psql."
        error_message "Sprawdź dane logowania, hosta, port oraz czy kontener Docker jest poprawnie uruchomiony i dostępny."
        exit 1
    fi
    
    unset PGPASSWORD
}

# --- Główna Egzekucja ---
main() {
    log_message "AI Command Center - Automatyzacja Konfiguracji Bazy Danych"
    echo "-----------------------------------------------------------------"
    
    check_dependencies
    echo "-----------------------------------------------------------------"
    
    setup_env_file
    echo "-----------------------------------------------------------------"
    
    create_compose_file
    echo "-----------------------------------------------------------------"
    
    start_containers
    echo "-----------------------------------------------------------------"
    
    verify_db_connection
    echo "-----------------------------------------------------------------"
    
    success_message "Środowisko bazy danych zostało pomyślnie skonfigurowane i uruchomione!"
    log_message "Następny krok: uruchom migracje Alembic, jeśli to konieczne:"
    log_message "cd backend/ && alembic upgrade head"
}

# Uruchomienie funkcji głównej
main
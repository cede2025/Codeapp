#!/bin/bash

# AI Command Center - Database Population & Migration Script
# Wersja: 1.1.0
# Autor: AI Assistant
# Opis: Ten skrypt należy uruchomić PO pomyślnym wykonaniu 'setup_database.sh'.
#       Automatyzuje on:
#       - Uruchomienie migracji Alembic w celu stworzenia/aktualizacji schematu bazy danych.
#       - Wypełnienie bazy danych danymi testowymi za pomocą skryptu 'seed.py'.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_DIR="backend"
readonly ENV_FILE="${BACKEND_DIR}/.env"
readonly VENV_PATH="${BACKEND_DIR}/.venv" # Dostosuj, jeśli używasz innej nazwy, np. venv
readonly SEED_SCRIPT_PATH="${BACKEND_DIR}/scripts/seed.py"

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
    read -p "$1 (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi
    return 0
}

# --- Główna Logika Skryptu ---

check_environment() {
    log_message "Weryfikacja środowiska..."
    
    if [ ! -f "${ENV_FILE}" ]; then
        error_message "Nie znaleziono pliku .env w katalogu '${BACKEND_DIR}/'. Uruchom najpierw 'setup_database.sh'."
        exit 1
    fi
    
    # Sprawdź, czy kontener Docker jest uruchomiony
    if ! docker ps | grep -q "aicc-postgres-db"; then
        error_message "Kontener Docker 'aicc-postgres-db' nie jest uruchomiony. Uruchom go za pomocą 'docker-compose up -d'."
        exit 1
    fi
    
    # Sprawdź, czy wirtualne środowisko istnieje i jest aktywowane
    if [ -z "${VIRTUAL_ENV}" ]; then
        if [ -d "${VENV_PATH}" ]; then
            warning_message "Wirtualne środowisko Pythona nie jest aktywowane."
            log_message "Próba aktywacji z '${VENV_PATH}/bin/activate'..."
            # shellcheck source=backend/.venv/bin/activate
            source "${VENV_PATH}/bin/activate"
            log_message "Wirtualne środowisko aktywowane."
        else
            error_message "Nie znaleziono wirtualnego środowiska w '${VENV_PATH}' i żadne nie jest aktywne."
            error_message "Utwórz je i zainstaluj zależności: 'python -m venv .venv' i 'pip install -r requirements.txt' w katalogu backend."
            exit 1
        fi
    fi
    
    if ! pip list | grep -q "alembic"; then
        error_message "Zależność 'alembic' nie jest zainstalowana w aktywnym środowisku wirtualnym."
        error_message "Uruchom 'pip install -r requirements.txt' w katalogu '${BACKEND_DIR}'."
        exit 1
    fi
    
    log_message "Środowisko zweryfikowane pomyślnie."
}

run_migrations() {
    log_message "Uruchamianie migracji bazy danych za pomocą Alembic..."
    cd "${BACKEND_DIR}"
    
    # Załaduj zmienne środowiskowe, aby Alembic miał dostęp do DATABASE_URL
    set -a # Automatycznie eksportuj zmienne
    # shellcheck source=.env
    source .env
    set +a
    
    log_message "Generowanie nowej rewizji migracji (jeśli są zmiany w modelach)..."
    # Używamy --autogenerate, aby wykryć zmiany w modelach
    alembic revision --autogenerate -m "Automated schema update check"
    
    log_message "Aplikowanie wszystkich migracji do bazy danych (upgrade to head)..."
    alembic upgrade head
    
    cd ..
    success_message "Migracje bazy danych zostały pomyślnie zastosowane."
}

run_seeding() {
    log_message "Wypełnianie bazy danych danymi testowymi..."
    
    if [ ! -f "${SEED_SCRIPT_PATH}" ]; then
        error_message "Nie znaleziono skryptu do wypełniania danych w '${SEED_SCRIPT_PATH}'."
        exit 1
    fi
    
    if confirm "Czy na pewno chcesz wypełnić bazę danych danymi testowymi? (Operacja jest bezpieczna do wielokrotnego uruchomienia)"; then
        # Uruchom skrypt Pythona w kontekście katalogu backend
        (cd "${BACKEND_DIR}" && python "scripts/seed.py")
        success_message "Baza danych została wypełniona danymi testowymi."
    else
        log_message "Pominięto wypełnianie bazy danych."
    fi
}

# --- Główna Egzekucja ---
main() {
    log_message "AI Command Center - Skrypt Migracji i Wypełniania Bazy Danych"
    echo "--------------------------------------------------------------------"
    
    check_environment
    echo "--------------------------------------------------------------------"
    
    run_migrations
    echo "--------------------------------------------------------------------"
    
    run_seeding
    echo "--------------------------------------------------------------------"
    
    success_message "Baza danych jest gotowa do pracy i testów!"
}

# Uruchomienie funkcji głównej
main
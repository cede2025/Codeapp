#!/bin/bash

# AI Command Center - Secrets Management Hardening Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje proces przenoszenia wszystkich wrażliwych danych
#       do zmiennych środowiskowych, usuwając domyślne wartości z kodu i
#       przygotowując aplikację do integracji z systemami zarządzania sekretami.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_PATH="backend"
readonly ENV_EXAMPLE_FILE="${BACKEND_PATH}/.env.example"
readonly CONFIG_PY_FILE="${BACKEND_PATH}/app/core/config.py"
readonly COMPOSE_FILE="docker-compose.yml"
readonly PROD_SECRETS_TEMPLATE="production.secrets.env.template"

# Lista kluczy wrażliwych
readonly SENSITIVE_KEYS=(
    "SECRET_KEY"
    "DB_PASSWORD"
    "REDIS_PASSWORD"
    "OPENROUTER_API_KEY"
    "EXTERNAL_SERVER_PIN_SALT"
    "XOR_ENCRYPTION_KEY"
)

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Secrets Management Hardening"
    echo "-----------------------------------------------------------------"
    
    if ! confirm "This script will modify core configuration files to enforce secrets management via environment variables. This is a one-way operation. Continue?"; then
        log_message "Operation cancelled by user."
        exit 0
    fi

    # Krok 1: Weryfikacja
    log_message "Verifying required files..."
    for file in "$ENV_EXAMPLE_FILE" "$CONFIG_PY_FILE" "$COMPOSE_FILE"; do
        if [ ! -f "$file" ]; then
            error_message "Required file not found: $file. Please ensure the project is fully set up."
            exit 1
        fi
    done
    success_message "File structure verified."

    # Krok 2: Sanityzacja .env.example
    log_message "Sanitizing '${ENV_EXAMPLE_FILE}' to remove default secrets..."
    cp "$ENV_EXAMPLE_FILE" "$ENV_EXAMPLE_FILE.bak"
    for key in "${SENSITIVE_KEYS[@]}"; do
        # Zastąp wartość po znaku równości placeholderem
        sed -i.tmp "s/^\(${key}\)=.*/\1=<your_${key,,}_here>/" "$ENV_EXAMPLE_FILE"
    done
    rm "${ENV_EXAMPLE_FILE}.tmp"
    success_message "Sanitized '${ENV_EXAMPLE_FILE}'. Previous version saved as .bak."

    # Krok 3: Utwardzenie config.py
    log_message "Hardening '${CONFIG_PY_FILE}' to require secrets from environment..."
    cp "$CONFIG_PY_FILE" "$CONFIG_PY_FILE.bak"
    for key in "${SENSITIVE_KEYS[@]}"; do
        # Usuń domyślną wartość, czyniąc pole wymaganym
        # Przykład: `SECRET_KEY: str = "default"` -> `SECRET_KEY: str`
        # Używamy `Optional` dla kluczy, które mogą być puste (np. REDIS_PASSWORD)
        if [[ "$key" == "REDIS_PASSWORD" ]]; then
             sed -i.tmp "s/^\(${key}: Optional\[str\]\) = .*/\1/" "$CONFIG_PY_FILE"
        else
             sed -i.tmp "s/^\(${key}: str\) = .*/\1/" "$CONFIG_PY_FILE"
        fi
    done
    rm "${CONFIG_PY_FILE}.tmp"
    success_message "Hardened '${CONFIG_PY_FILE}'. Secrets are now mandatory environment variables."

    # Krok 4: Modyfikacja docker-compose.yml
    log_message "Updating '${COMPOSE_FILE}' to demonstrate production-like secret injection..."
    cp "$COMPOSE_FILE" "$COMPOSE_FILE.bak"
    # Użyj awk do skomplikowanej edycji: zakomentuj env_file i dodaj sekcję environment
    awk '
    /env_file:/ {
        print "    # For development, you can use env_file to load secrets from backend/.env"
        print "    # env_file:"
        print "    #   - ./backend/.env"
        print ""
        print "    # For production, it is recommended to pass secrets directly as environment variables"
        print "    # Example (uncomment and set these in your shell or CI/CD environment):"
        print "    # environment:"
        print "    #   - DB_PASSWORD=\${DB_PASSWORD}"
        print "    #   - SECRET_KEY=\${SECRET_KEY}"
        print "    #   - OPENROUTER_API_KEY=\${OPENROUTER_API_KEY}"
        next
    }
    /- \.\/backend\/.env/ { next }
    { print }
    ' "$COMPOSE_FILE.bak" > "$COMPOSE_FILE"
    success_message "Updated '${COMPOSE_FILE}' with production secret management patterns."

    # Krok 5: Stworzenie szablonu sekretów produkcyjnych
    log_message "Generating production secrets template at '${PROD_SECRETS_TEMPLATE}'..."
    {
        echo "# Production Secrets Template for AI Command Center"
        echo "# This file lists all mandatory secrets required to run the application in production."
        echo "# DO NOT commit this file with real values. Use it as a reference for your secrets management system."
        echo "# (e.g., AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets)"
        echo ""
        echo "# Generated on: $(date)"
        echo ""
        for key in "${SENSITIVE_KEYS[@]}"; do
            echo "${key}="
        done
    } > "$PROD_SECRETS_TEMPLATE"
    success_message "Created production secrets template."

    # Krok 6: Instrukcje końcowe
    echo
    warning_message "------------------------- CRITICAL: REVIEW CHANGES -------------------------"
    warning_message "The application has been hardened for secure secret management."
    echo
    echo -e "  \033[1;33m1. Your code is now more secure:\033[0m"
    echo -e "     - The application will FAIL TO START if any secret (like OPENROUTER_API_KEY) is not set."
    echo -e "     - Default, weak secrets have been removed from the codebase."
    echo
    echo -e "  \033[1;33m2. Your Development Workflow:\033[0m"
    echo -e "     - All secrets MUST now be defined in your local \033[1;32m'backend/.env'\033[0m file."
    echo -e "     - This file is git-ignored and should NEVER be committed."
    echo -e "     - To run with Docker Compose, uncomment the 'env_file' section in 'docker-compose.yml'."
    echo
    echo -e "  \033[1;33m3. Preparing for Production:\033[0m"
    echo -e "     - Use the new \033[1;32m'${PROD_SECRETS_TEMPLATE}'\033[0m file as a checklist for your secrets manager."
    echo -e "     - In your production environment (e.g., Kubernetes, AWS), inject these variables into the containers."
    warning_message "--------------------------------------------------------------------------"
    echo
    success_message "Secrets management hardening complete!"
}

# Uruchomienie funkcji głównej
main
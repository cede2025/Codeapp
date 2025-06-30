#!/bin/bash

# AI Command Center - Real Celery Integration Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje integrację Celery z aplikacją, zastępując
#       symulacje prawdziwym, asynchronicznym systemem zadań opartym na Redis.
#       Modyfikuje konfigurację, logikę API i tworzy punkty wejścia dla workera.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_PATH="backend"
readonly APP_PATH="${BACKEND_PATH}/app"
readonly REQUIREMENTS_FILE="${BACKEND_PATH}/requirements.txt"
readonly ENV_FILE="${BACKEND_PATH}/.env"
readonly CONFIG_PY_FILE="${APP_PATH}/core/config.py"
readonly CELERY_APP_FILE="${APP_PATH}/tasks/celery_app.py"
readonly CELERY_TASKS_FILE="${APP_PATH}/tasks/celery_tasks.py"
readonly ROUTER_TASKS_FILE="${APP_PATH}/api/v1/router_tasks.py"
readonly RUN_WORKER_SCRIPT="${BACKEND_PATH}/scripts/run_celery_worker.sh"

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# --- Definicje Kodu do Wstrzyknięcia ---

read -r -d '' REAL_CELERY_APP_CODE <<'EOF'
# backend/app/tasks/celery_app.py

from celery import Celery
from ..core.config import settings

celery_app = Celery(
   "ai_command_center_worker",
   broker=settings.CELERY_BROKER_URL,
   backend=settings.CELERY_RESULT_BACKEND,
   include=[
       'app.tasks.celery_tasks'  # Ścieżka do modułu z zadaniami
   ]
)

celery_app.conf.update(
   task_serializer='json',
   result_serializer='json',
   accept_content=['json'],
   timezone='UTC',
   enable_utc=True,
   result_expires=3600, # Czas przechowywania wyników (w sekundach)
   task_track_started=True,
)

if __name__ == '__main__':
   celery_app.start()
EOF

read -r -d '' REAL_CELERY_TASK_CODE <<'EOF'
# backend/app/tasks/celery_tasks.py

import asyncio
from .celery_app import celery_app
from ..core.ai_agent import MainAIAgent
from ..core.openrouter_client import OpenRouterClient
from ..models_schemas.ai_core import AITaskDefinition, AITaskResult

# Wzorzec fabryki, aby unikać globalnego stanu i poprawnie zarządzać zależnościami w workerze
def get_agent_instance() -> MainAIAgent:
    """Creates and returns an instance of the MainAIAgent."""
    # Ta funkcja jest wywoływana wewnątrz procesu workera,
    # więc tworzy nowe instancje klientów dla każdego zadania (lub per proces workera).
    client = OpenRouterClient()
    # W przyszłości można tu wstrzyknąć inne zależności, np. sesję DB.
    agent = MainAIAgent(openrouter_client=client)
    return agent

@celery_app.task(name="tasks.orchestrate_ai_lifecycle")
async def orchestrate_ai_lifecycle_task(task_def_dict: dict) -> dict:
    """
    Asynchronous Celery task to run the full AI orchestration lifecycle.
    """
    agent = get_agent_instance()
    task_definition = AITaskDefinition(**task_def_dict)
    
    result: AITaskResult = await agent.orchestrate_task(task_definition)
    
    # Zwracamy wynik jako słownik, aby był serializowalny do JSON
    return result.model_dump()
EOF

read -r -d '' REAL_API_ROUTER_CODE <<'EOF'
# backend/app/api/v1/router_tasks.py

from fastapi import APIRouter, Body, Depends, HTTPException, Path
from celery.result import AsyncResult

from ...tasks.celery_app import celery_app
from ...tasks.celery_tasks import orchestrate_ai_lifecycle_task
from ...models_schemas.ai_core import AITaskDefinition
from ...models_schemas.responses.task_responses import TaskDelegationResult, TaskStatusResponse

router = APIRouter()

@router.post(
    "/execute",
    response_model=TaskDelegationResult,
    status_code=202, # Accepted
    summary="Trigger an AI Task for Asynchronous Execution"
)
async def trigger_ai_task(
    task_definition: AITaskDefinition,
):
    """
    Accepts a task definition and queues it for background processing via Celery.
    Immediately returns a task ID for status tracking.
    """
    # Użyj .delay() do uruchomienia zadania w tle.
    # Przekazujemy definicję jako słownik, bo Celery serializuje argumenty.
    task = orchestrate_ai_lifecycle_task.delay(task_definition.model_dump())
    
    return TaskDelegationResult(
        task_id=task_definition.task_id,
        celery_task_id=task.id,
        status="QUEUED",
        message="Task has been accepted and queued for execution."
    )

@router.get(
    "/status/{celery_task_id}",
    response_model=TaskStatusResponse,
    summary="Check the Status and Result of an Asynchronous Task"
)
async def get_task_status(
    celery_task_id: str = Path(..., description="The ID of the Celery task returned by /execute")
):
    """
    Retrieves the status and result of a background task.
    Poll this endpoint until the status is 'SUCCESS' or 'FAILURE'.
    """
    task_result = AsyncResult(celery_task_id, app=celery_app)
    
    response_data = {
        "celery_task_id": celery_task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None,
    }
    
    if task_result.failed():
        # W przypadku błędu, wynik zawiera wyjątek
        response_data["traceback"] = task_result.traceback

    return response_data
EOF

read -r -d '' RUN_CELERY_WORKER_SCRIPT_CODE <<'EOF'
#!/bin/bash
# AI Command Center - Celery Worker Runner Script

set -e

# Przejdź do katalogu nadrzędnego (backend), aby ścieżki importu działały poprawnie
cd "$(dirname "$0")/.."

echo "🚀 Starting AI Command Center Celery Worker..."
echo "----------------------------------------------------"
echo "Logs will be shown below. Press Ctrl+C to stop."
echo "Ensure Redis container is running."
echo "----------------------------------------------------"

# Uruchomienie workera Celery
# -A: Aplikacja Celery
# -l: Poziom logowania
# -P: Pula workerów. 'gevent' lub 'eventlet' są zalecane dla zadań I/O-bound (async).
#     'solo' jest dobre do debugowania.
celery -A app.tasks.celery_app worker -l info -P gevent
EOF

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Real Celery Integration"
    echo "-----------------------------------------------------------------"
    
    if ! confirm "This script will integrate real Celery, modifying core files. It's recommended to have a backup. Continue?"; then
        log_message "Operation cancelled by user."
        exit 0
    fi

    # Krok 1: Aktualizacja zależności
    log_message "Updating '${REQUIREMENTS_FILE}'..."
    if ! grep -q "celery" "$REQUIREMENTS_FILE"; then
        # Dodaj Celery z obsługą Redis
        sed -i'' '/# Asynchronous Task Queue/a \
celery[redis]==5.3.6' "$REQUIREMENTS_FILE"
        success_message "Added 'celery[redis]' to requirements.txt."
    else
        warning_message "'celery' already exists in requirements.txt. Skipping."
    fi

    # Krok 2: Konfiguracja środowiska
    log_message "Configuring '${ENV_FILE}' and '${CONFIG_PY_FILE}' for Redis..."
    # Upewnij się, że zmienne Redis są w .env
    if ! grep -q "REDIS_HOST" "$ENV_FILE"; then
        echo -e "\n# Redis Configuration\nREDIS_HOST=\"localhost\"\nREDIS_PORT=\"6379\"\nREDIS_DB_CELERY=\"0\"\nREDIS_DB_CACHE=\"1\"" >> "$ENV_FILE"
    fi
    # Upewnij się, że zmienne Celery są w .env
    if ! grep -q "CELERY_BROKER_URL" "$ENV_FILE"; then
        echo -e "\nCELERY_BROKER_URL=\"redis://\${REDIS_HOST}:\${REDIS_PORT}/\${REDIS_DB_CELERY}\"" >> "$ENV_FILE"
        echo "CELERY_RESULT_BACKEND=\"redis://\${REDIS_HOST}:\${REDIS_PORT}/\${REDIS_DB_CELERY}\"" >> "$ENV_FILE"
    fi
    # Upewnij się, że config.py poprawnie konstruuje URL-e
    if ! grep -q "if self.CELERY_BROKER_URL" "$CONFIG_PY_FILE"; then
        # Dodaj logikę konstruowania URL-i do config.py
        sed -i'' '/model_config = SettingsConfigDict/i \
\
    def __init__(self, **values):\
        super().__init__(**values)\
        redis_auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""\
        if "redis://localhost" in self.CELERY_BROKER_URL:\
            self.CELERY_BROKER_URL = f"redis://{redis_auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_CELERY}"\
        if "redis://localhost" in self.CELERY_RESULT_BACKEND:\
            self.CELERY_RESULT_BACKEND = f"redis://{redis_auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_CELERY}"\
' "$CONFIG_PY_FILE"
    fi
    success_message "Environment configured for Celery with Redis."

    # Krok 3: Wstrzyknięcie kodu Celery
    log_message "Injecting real Celery implementation..."
    # Kopia zapasowa i zastąpienie plików
    cp "$CELERY_APP_FILE" "$CELERY_APP_FILE.bak"
    echo "$REAL_CELERY_APP_CODE" > "$CELERY_APP_FILE"
    
    cp "$CELERY_TASKS_FILE" "$CELERY_TASKS_FILE.bak"
    echo "$REAL_CELERY_TASK_CODE" > "$CELERY_TASKS_FILE"
    
    cp "$ROUTER_TASKS_FILE" "$ROUTER_TASKS_FILE.bak"
    echo "$REAL_API_ROUTER_CODE" > "$ROUTER_TASKS_FILE"
    
    success_message "Replaced Celery and API Task files with production-ready versions."

    # Krok 4: Utworzenie skryptu startowego workera
    log_message "Creating Celery worker startup script..."
    mkdir -p "$(dirname "$RUN_WORKER_SCRIPT")"
    echo "$RUN_CELERY_WORKER_SCRIPT_CODE" > "$RUN_WORKER_SCRIPT"
    chmod +x "$RUN_WORKER_SCRIPT"
    success_message "Created worker script at '${RUN_WORKER_SCRIPT}'."

    # Krok 5: Instrukcje końcowe
    echo
    warning_message "------------------------- ACTION REQUIRED -------------------------"
    warning_message "Real Celery integration is complete. Follow these steps to run:"
    echo
    echo -e "  \033[1;33m1. Install new dependencies:\033[0m"
    echo -e "     cd ${BACKEND_PATH} && pip install -r requirements.txt"
    echo
    echo -e "  \033[1;33m2. Start infrastructure (PostgreSQL & Redis):\033[0m"
    echo -e "     docker-compose up -d"
    echo
    echo -e "  \033[1;33m3. In a NEW terminal, start the FastAPI server:\033[0m"
    echo -e "     cd ${BACKEND_PATH} && uvicorn app.main:app --reload"
    echo
    echo -e "  \033[1;33m4. In ANOTHER new terminal, start the Celery worker:\033[0m"
    echo -e "     ./${RUN_WORKER_SCRIPT}"
    echo
    echo -e "  \033[1;33m5. Test the new API flow:\033[0m"
    echo -e "     a) Send a \033[1;32mPOST\033[0m request to \033[1;34m/api/v1/tasks/execute\033[0m. You will get a 'celery_task_id'."
    echo -e "     b) Send a \033[1;32mGET\033[0m request to \033[1;34m/api/v1/tasks/status/{celery_task_id}\033[0m to check the result."
    warning_message "-----------------------------------------------------------------"
    echo
    success_message "Integration with real Celery is complete!"
}

# Uruchomienie funkcji głównej
main
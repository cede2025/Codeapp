#!/bin/bash

# AI Command Center - Real API Client Integration Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt finalizuje aplikację, zastępując symulowany klient OpenRouter
#       prawdziwym, asynchronicznym klientem HTTP (httpx), ulepszając schematy
#       Pydantic i integrując wszystko z cyklem życia aplikacji FastAPI.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_PATH="backend"
readonly APP_PATH="${BACKEND_PATH}/app"
readonly REQUIREMENTS_FILE="${BACKEND_PATH}/requirements.txt"
readonly OPENROUTER_CLIENT_FILE="${APP_PATH}/core/openrouter_client.py"
readonly OPENROUTER_RESPONSES_FILE="${APP_PATH}/models_schemas/responses/openrouter_responses.py"
readonly MAIN_PY_FILE="${APP_PATH}/main.py"

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# --- Definicje Kodu do Wstrzyknięcia ---

read -r -d '' REAL_OPENROUTER_CLIENT_CODE <<'EOF'
# backend/app/core/openrouter_client.py

import httpx
import logging
import time
from typing import List, Dict, Any, Optional

from ..core.config import settings
from ..models_schemas.responses.openrouter_responses import ModelListResponse, ModelChoiceResponse, ChatCompletionApiResponse

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    A production-ready, asynchronous client for interacting with the OpenRouter.ai API.
    """
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self,
                 api_key: str = settings.OPENROUTER_API_KEY,
                 site_url: str = settings.OPENROUTER_SITE_URL,
                 app_name: str = settings.OPENROUTER_APP_NAME,
                 model_cache_ttl: int = settings.OPENROUTER_MODEL_CACHE_TTL_SECONDS):
        
        if not api_key:
            logger.error("OpenRouter API key is required but not provided.")
            raise ValueError("OpenRouter API key is required. Please set OPENROUTER_API_KEY in your .env file.")
        
        self.api_key = api_key
        self.model_cache_ttl = model_cache_ttl
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url or "",
            "X-Title": app_name or "",
        }
        
        self._available_models: List[ModelChoiceResponse] = []
        self._models_last_fetched_time: float = 0.0
        self._async_client = httpx.AsyncClient(headers=headers, timeout=60.0)
        logger.info("Real OpenRouterClient initialized.")

    async def close(self):
        """Closes the underlying HTTPX client. Should be called during application shutdown."""
        if not self._async_client.is_closed:
            await self._async_client.aclose()
            logger.info("OpenRouterClient's HTTP session closed.")

    async def get_available_models(self, force_refresh: bool = False) -> List[ModelChoiceResponse]:
        """
        Fetches and caches the list of available models from OpenRouter.
        """
        current_time = time.time()
        if not force_refresh and self._available_models and \
           (current_time - self._models_last_fetched_time < self.model_cache_ttl):
            logger.debug("Returning cached OpenRouter models.")
            return self._available_models

        logger.info(f"Fetching available models from OpenRouter (Force refresh: {force_refresh})...")
        try:
            response = await self._async_client.get(f"{self.BASE_URL}/models")
            response.raise_for_status()
            
            parsed_response = ModelListResponse(**response.json())
            self._available_models = parsed_response.data
            self._models_last_fetched_time = current_time
            logger.info(f"Successfully fetched and cached {len(self._available_models)} models.")
            return self._available_models
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching models: {e.response.status_code} - {e.response.text}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error processing models response: {e}", exc_info=True)
        
        return self._available_models if self._available_models else []

    async def chat_completion(self, model_id: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Performs a chat completion request to OpenRouter.
        """
        logger.debug(f"Requesting chat completion from model '{model_id}'.")
        payload = {"model": model_id, "messages": messages, **kwargs}

        try:
            response = await self._async_client.post(f"{self.BASE_URL}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"OpenRouter API error for model '{model_id}': {error_detail}")
            return {"error": {"message": error_detail, "model_id": model_id}}
        except httpx.RequestError as e:
            logger.error(f"Network error during chat completion with '{model_id}': {e}", exc_info=True)
            return {"error": {"message": f"Network request failed: {e}", "model_id": model_id}}
        except Exception as e:
            logger.error(f"Unexpected error during chat completion: {e}", exc_info=True)
            return {"error": {"message": f"An unexpected error occurred: {e}", "model_id": model_id}}
EOF

read -r -d '' REAL_OPENROUTER_RESPONSES_CODE <<'EOF'
# backend/app/models_schemas/responses/openrouter_responses.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ModelChoiceResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    pricing: Dict[str, str]
    context_length: Optional[int] = None
    architecture: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class ModelListResponse(BaseModel):
    data: List[ModelChoiceResponse]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[float] = None

class ChatCompletionApiResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
EOF

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Real API Client Integration"
    echo "-----------------------------------------------------------------"
    
    if ! confirm "This script will replace simulated components with a real API client. It's recommended to have a backup. Continue?"; then
        log_message "Operation cancelled by user."
        exit 0
    fi

    # Krok 1: Weryfikacja
    log_message "Verifying file structure..."
    for file in "$REQUIREMENTS_FILE" "$OPENROUTER_CLIENT_FILE" "$OPENROUTER_RESPONSES_FILE" "$MAIN_PY_FILE"; do
        if [ ! -f "$file" ]; then
            error_message "Required file not found: $file. Please ensure the project structure is correct."
            exit 1
        fi
    done
    success_message "File structure verified."

    # Krok 2: Aktualizacja zależności
    log_message "Updating '${REQUIREMENTS_FILE}'..."
    if ! grep -q "httpx" "$REQUIREMENTS_FILE"; then
        echo "httpx==0.27.0" >> "$REQUIREMENTS_FILE"
        success_message "Added 'httpx' to requirements.txt."
    else
        warning_message "'httpx' already exists in requirements.txt. Skipping."
    fi

    # Krok 3: Zastąpienie plików
    log_message "Replacing simulated files with production-ready versions..."
    
    # Kopia zapasowa i zastąpienie klienta
    cp "$OPENROUTER_CLIENT_FILE" "$OPENROUTER_CLIENT_FILE.bak"
    echo "$REAL_OPENROUTER_CLIENT_CODE" > "$OPENROUTER_CLIENT_FILE"
    success_message "Replaced '${OPENROUTER_CLIENT_FILE}' with real implementation."
    
    # Kopia zapasowa i zastąpienie schematów odpowiedzi
    # Sprawdź, czy plik istnieje, zanim go skopiujesz
    if [ -f "$OPENROUTER_RESPONSES_FILE" ]; then
        cp "$OPENROUTER_RESPONSES_FILE" "$OPENROUTER_RESPONSES_FILE.bak"
    fi
    echo "$REAL_OPENROUTER_RESPONSES_CODE" > "$OPENROUTER_RESPONSES_FILE"
    success_message "Replaced/Created '${OPENROUTER_RESPONSES_FILE}' with detailed schemas."

    # Krok 4: Modyfikacja main.py
    log_message "Updating '${MAIN_PY_FILE}' to manage the real client's lifecycle..."
    cp "$MAIN_PY_FILE" "$MAIN_PY_FILE.bak"
    # Użyj awk do bezpiecznego zastąpienia bloku lifespan
    awk '
        BEGIN { p = 1 }
        /@asynccontextmanager/ {
            print;
            print "async def lifespan(app: FastAPI):";
            print "    # Startup";
            print "    logger.info(f\"Starting {settings.APP_NAME} v{settings.APP_VERSION} in {settings.ENVIRONMENT} mode.\")";
            print "";
            print "    # Initialize OpenRouter client";
            print "    app_lifespan_globals[\"openrouter_client\"] = OpenRouterClient()";
            print "    asyncio.create_task(app_lifespan_globals[\"openrouter_client\"].get_available_models(force_refresh=True)) # Pre-fetch models";
            print "    logger.info(\"Real OpenRouter client initialized.\")";
            print "";
            print "    # Initialize other services that might depend on the client...";
            print "    # (Example: MainAIAgent)";
            print "    app_lifespan_globals[\"main_ai_agent\"] = MainAIAgent(openrouter_client=app_lifespan_globals[\"openrouter_client\"])";
            print "    logger.info(\"Main AI Agent initialized.\")";
            print "";
            print "    yield # Application runs here";
            print "";
            print "    # Shutdown";
            print "    logger.info(f\"Shutting down {settings.APP_NAME}...\")";
            print "    await app_lifespan_globals[\"openrouter_client\"].close()";
            print "    logger.info(\"Shutdown complete.\")";
            p = 0;
            next;
        }
        /yield/ { p = 1; next }
        p { print }
    ' "$MAIN_PY_FILE.bak" > "$MAIN_PY_FILE"
    # Usuń stare, niepotrzebne już zależności DI, jeśli istnieją
    sed -i'' -e '/app\.dependency_overrides\[MainAIAgent\]/d' \
            -e '/app\.dependency_overrides\[OpenRouterClient\]/d' \
            "$MAIN_PY_FILE"
    success_message "Updated '${MAIN_PY_FILE}' with real client lifecycle management."

    # Krok 5: Instrukcje końcowe
    echo
    warning_message "------------------------- ACTION REQUIRED -------------------------"
    warning_message "The application is now configured to use the real OpenRouter API."
    echo
    echo -e "  \033[1;33m1. Install new dependencies:\033[0m"
    echo -e "     cd ${BACKEND_PATH} && pip install -r requirements.txt"
    echo
    echo -e "  \033[1;33m2. Configure your API Key:\033[0m"
    echo -e "     Open the file '${BACKEND_PATH}/.env' and set a valid key for:"
    echo -e "     \033[1;32mOPENROUTER_API_KEY=\"your_real_openrouter_api_key_here\"\033[0m"
    echo
    warning_message "The application will not start correctly without a valid API key."
    warning_message "-----------------------------------------------------------------"
    echo
    success_message "Integration with the real world is complete!"
}

# Uruchomienie funkcji głównej
main
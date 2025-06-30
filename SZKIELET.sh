#!/bin/bash

# AI Command Center - Termux Advanced Assembly Script
# Wersja: 1.1.0 (Złożona)
# Autor: Eryk D z pomocą AI
# Data: $(date +%Y-%m-%d)
# Opis: Ten skrypt automatyzuje tworzenie struktury projektu AI Command Center,
#              włączając w to SZCZEGÓŁOWE pliki placeholderów i szkielety komponentów dla backendu,
#              frontendu (z motywem Cyberpunk), zaawansowanych funkcji, dokumentacji i konfiguracji.
#              Generuje archiwum ZIP gotowe do pracy w Visual Studio Code.
#
# OGRANICZENIE: Ten skrypt generuje bardzo szczegółowy SZKIELET i PROJEKT.
#             Pełna implementacja wszystkich funkcji korporacyjnych wymaga znacznego
#             wysiłku deweloperskiego wykraczającego poza zakres tego skryptu generującego.

# --- Konfiguracja Skryptu ---
set -euo pipefail # Zakończ w przypadku błędu, traktuj niezainicjowane zmienne jako błąd, pipefail

# --- Zmienne Projektowe ---
readonly PROJECT_NAME="AI_Command_Center_Advanced_VSCode_Ready"
readonly BUILD_DIR_BASE="build_temp_aic_adv" # Podstawowa nazwa dla katalogu budowy
readonly ZIP_FILENAME="${PROJECT_NAME}.zip"
readonly SCRIPT_VERSION="1.1.0"

# --- 0Funkcje Pomocnicze ---
log_message() {
   echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"
}

error_message() {
   echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2
}

check_dependencies() {
   log_message "Sprawdzanie zależności..."
   local missing_deps=0
   for cmd in zip curl mktemp readlink; do # Dodano mktemp i readlink dla większej niezawodności
       if ! command -v "$cmd" &> /dev/null; then
           error_message "Zależność '$cmd' nie została znaleziona. Proszę ją zainstalować (np. 'pkg install $cmd' lub 'apt install $cmd')."
           missing_deps=1
       fi
   done
   if [ "$missing_deps" -eq 1 ]; then
       error_message "Proszę zainstalować brakujące zależności i spróbować ponownie."
       exit 1
   fi
   log_message "Wszystkie zależności (zip, curl, mktemp, readlink) są obecne."
}

# --- Funkcje Tworzenia Katalogów i Plików ---

create_project_structure() {
   log_message "Tworzenie głównego katalogu projektu: ${BUILD_DIR}/${PROJECT_NAME}"
   mkdir -p "${BUILD_DIR}/${PROJECT_NAME}"

   log_message "Tworzenie głównych podkatalogów..."
   cd "${BUILD_DIR}/${PROJECT_NAME}"
   mkdir -p backend frontend docs scripts .vscode config
   mkdir -p frontend/assets # Przeniesiono tworzenie assets tutaj dla jasności
   cd .. # Powrót do BUILD_DIR
   log_message "Stworzono podstawową strukturę projektu."
}

create_backend_files() {
   log_message "Tworzenie SZCZEGÓŁOWEJ struktury i plików backendu (FastAPI + Celery)..."
   local backend_path="${BUILD_DIR}/${PROJECT_NAME}/backend"
   mkdir -p "${backend_path}/app/core"
   mkdir -p "${backend_path}/app/api/v1"
   mkdir -p "${backend_path}/app/services/discovery" # Podpakiet dla discovery
   mkdir -p "${backend_path}/app/services/orchestration"
   mkdir -p "${backend_path}/app/tasks"
   mkdir -p "${backend_path}/app/models_schemas/requests"
   mkdir -p "${backend_path}/app/models_schemas/responses"
   mkdir -p "${backend_path}/app/models_schemas/db" # Dla modeli ORM, jeśli używane
   mkdir -p "${backend_path}/app/security"
   mkdir -p "${backend_path}/app/utils"
   mkdir -p "${backend_path}/app/websockets"
   mkdir -p "${backend_path}/app/monitoring"
   mkdir -p "${backend_path}/tests/api"
   mkdir -p "${backend_path}/tests/services"
   mkdir -p "${backend_path}/tests/tasks"

   # requirements.txt (bardziej kompleksowe)
   cat << EOF > "${backend_path}/requirements.txt"
# Core FastAPI & Web Server
fastapi==0.110.0
uvicorn[standard]==0.27.1
python-dotenv==1.0.1
pydantic==2.6.3
pydantic-settings==2.2.1 # For settings management

# Asynchronous Task Queue
celery==5.3.6
redis==5.0.1 # For Celery broker/backend & caching

# Networking & API Interaction
requests==2.31.0 # For OpenRouter client and external server health checks
httpx==0.27.0 # Modern async HTTP client, good alternative to requests for async code

# WebSockets
websockets==12.0

# Security
cryptography==42.0.5 # For encryption, hashing
passlib[bcrypt]==1.7.4 # For password hashing (e.g., server PINs)
python-jose[cryptography]==3.3.0 # For JWT tokens if implementing enterprise auth

# Monitoring & Observability
prometheus-client==0.19.0 # For Prometheus metrics
# opentelemetry-api==1.22.0
# opentelemetry-sdk==1.22.0
# opentelemetry-instrumentation-fastapi==0.43b0
# opentelemetry-instrumentation-celery==0.43b0
# opentelemetry-instrumentation-requests==0.43b0
# opentelemetry-exporter-otlp-proto-http==1.22.0 # Or other exporters

# Database (Example: SQLAlchemy for PostgreSQL - uncomment if needed)
# sqlalchemy[asyncio]==2.0.25
# asyncpg==0.29.0 # Async driver for PostgreSQL
# alembic==1.13.1 # For database migrations

# Machine Learning (for intelligent routing - placeholder, specific libraries depend on model)
# scikit-learn # Example for general ML tasks
# tensorflow # or pytorch, etc.

# Utilities
# structlog # For structured logging
# email-validator # For validating email addresses if needed
EOF

   # .env.example (bardziej szczegółowe)
   cat << EOF > "${backend_path}/.env.example"
# Backend Application Configuration
APP_NAME="AI Command Center Backend"
APP_VERSION="${SCRIPT_VERSION}"
DEBUG=True
ENVIRONMENT="development" # development, staging, production
SECRET_KEY="your_very_strong_random_secret_key_for_jwt_and_other_crypto"
ALLOWED_HOSTS="localhost,127.0.0.1" # Comma-separated

# CORS Origins (comma-separated, use '*' for development only)
CORS_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000" # Adjust for Flutter dev port

# Database (Example: PostgreSQL - uncomment and configure if using SQL DB)
# DB_ENGINE="postgresql+asyncpg"
# DB_USER="aicc_user"
# DB_PASSWORD="aicc_password"
# DB_HOST="localhost"
# DB_PORT="5432"
# DB_NAME="aicc_db"
# DATABASE_URL="\${DB_ENGINE}://\${DB_USER}:\${DB_PASSWORD}@\${DB_HOST}:\${DB_PORT}/\${DB_NAME}"

# Redis Configuration
REDIS_HOST="localhost"
REDIS_PORT="6379"
REDIS_DB_CELERY="0"
REDIS_DB_CACHE="1"
REDIS_PASSWORD="" # Leave empty if no password

CELERY_BROKER_URL="redis://\${REDIS_HOST}:\${REDIS_PORT}/\${REDIS_DB_CELERY}"
CELERY_RESULT_BACKEND="redis://\${REDIS_HOST}:\${REDIS_PORT}/\${REDIS_DB_CELERY}"

# OpenRouter Configuration
OPENROUTER_API_KEY="your_openrouter_api_key_here"
OPENROUTER_SITE_URL="https://youraicommancenter.com" # For HTTP-Referer
OPENROUTER_APP_NAME="AICommandCenter" # For X-Title
OPENROUTER_DEFAULT_MODEL="anthropic/claude-3-sonnet-20240229" # Example
OPENROUTER_MODEL_CACHE_TTL_SECONDS=3600

# External Server Communication
EXTERNAL_SERVER_PIN_SALT="another_strong_random_salt_for_pin_hashing"
XOR_ENCRYPTION_KEY="a_simple_key_for_xor_example_use_aes_in_prod" # Replace with AES key for production

# Monitoring
PROMETHEUS_ENABLED=True

# Token Management & Budgeting
DEFAULT_TOKEN_BUDGET_USD=100.00
BUDGET_WARNING_THRESHOLD_PERCENT=80
EOF

   # Dockerfile
   cat << EOF > "${backend_path}/Dockerfile"
# Dockerfile for FastAPI Backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any (e.g., for psycopg2-binary, or build tools for some libs)
# RUN apt-get update && apt-get install -y build-essential libpq-dev ... && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Consider using --no-cache-dir for smaller images, but might be slower for rebuilds
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Expose port (ensure it matches Uvicorn config)
EXPOSE 8000

# Command to run the application (adjust app.main:app path if needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

   # app/__init__.py
   touch "${backend_path}/app/__init__.py"

   # app/main.py
   cat << EOF > "${backend_path}/app/main.py"
#!/usr/bin/env python3
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import settings
from .api.v1 import (
   router_openrouter,
   router_servers,
   router_tasks,
   router_system, # For system health, schema evolution
)
from .core.ai_agent import MainAIAgent
from .core.openrouter_client import OpenRouterClient
from .services.discovery.server_registry import ServerRegistry
from .services.discovery.connection_manager import ConnectionManager
from .services.orchestration.load_balancer import TaskLoadBalancer
from .websockets.manager import websocket_manager
# from .security import auth # For enterprise authentication if implemented
from .monitoring.prometheus import PrometheusMonitor # Placeholder

# Configure logging
logging.basicConfig(level=logging.INFO if settings.ENVIRONMENT != "development" else logging.DEBUG)
logger = logging.getLogger(__name__)

# Global instances (consider dependency injection for more complex scenarios)
# These would typically be initialized in the lifespan manager
app_lifespan_globals = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
   # Startup
   logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} in {settings.ENVIRONMENT} mode.")

   # Initialize OpenRouter client
   app_lifespan_globals["openrouter_client"] = OpenRouterClient(
       api_key=settings.OPENROUTER_API_KEY,
       site_url=settings.OPENROUTER_SITE_URL,
       app_name=settings.OPENROUTER_APP_NAME,
       model_cache_ttl=settings.OPENROUTER_MODEL_CACHE_TTL_SECONDS
   )
   await app_lifespan_globals["openrouter_client"].refresh_available_models() # Initial fetch
   logger.info("OpenRouter client initialized and models fetched.")

   # Initialize Server Discovery services
   app_lifespan_globals["server_registry"] = ServerRegistry() # TODO: Add persistent storage
   app_lifespan_globals["connection_manager"] = ConnectionManager(app_lifespan_globals["server_registry"])
   # await app_lifespan_globals["connection_manager"].start_monitoring() # Start background health checks
   logger.info("Server Discovery services initialized.")

   # Initialize Task Load Balancer
   app_lifespan_globals["task_load_balancer"] = TaskLoadBalancer(
       server_registry=app_lifespan_globals["server_registry"],
       openrouter_client=app_lifespan_globals["openrouter_client"]
   )
   logger.info("Task Load Balancer initialized.")

   # Initialize Main AI Agent
   app_lifespan_globals["main_ai_agent"] = MainAIAgent(
       openrouter_client=app_lifespan_globals["openrouter_client"],
       task_load_balancer=app_lifespan_globals["task_load_balancer"],
       # TODO: Pass other necessary dependencies like DB access, Celery app
   )
   logger.info("Main AI Agent initialized.")

   # Initialize Prometheus Monitor if enabled
   if settings.PROMETHEUS_ENABLED:
       app_lifespan_globals["prometheus_monitor"] = PrometheusMonitor()
       app.add_route("/metrics", app_lifespan_globals["prometheus_monitor"].handle_metrics)
       logger.info("Prometheus metrics endpoint enabled at /metrics.")

   # TODO: Initialize Database connections if using SQL DB
   # TODO: Initialize Celery app instance if not managed globally elsewhere

   yield # Application runs here

   # Shutdown
   logger.info(f"Shutting down {settings.APP_NAME}...")
   # await app_lifespan_globals["connection_manager"].stop_monitoring()
   # TODO: Gracefully close DB connections, WebSocket connections, etc.
   logger.info("Shutdown complete.")


app = FastAPI(
   title=settings.APP_NAME,
   version=settings.APP_VERSION,
   description="Enterprise-grade backend for AI task orchestration, OpenRouter integration, and external server management. Features advanced capabilities including intelligent task routing, real-time monitoring, and dynamic model management.",
   lifespan=lifespan,
   # root_path="/api/v1" # If serving under a prefix like /api/v1 globally
   # TODO: Add OpenAPI tags metadata for better docs
)

# Custom Exception Handler for RequestValidationError for more structured errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
   return JSONResponse(
       status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
       content={"detail": exc.errors(), "body": exc.body if hasattr(exc, 'body') else None},
   )

# Global Exception Handler (example)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
   logger.error(f"Unhandled exception for request {request.url}: {exc}", exc_info=True)
   return JSONResponse(
       status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
       content={"detail": "An unexpected internal server error occurred."},
   )

# CORS Middleware
if settings.CORS_ALLOWED_ORIGINS:
   origins = [origin.strip() for origin in settings.CORS_ALLOWED_ORIGINS.split(",")]
   app.add_middleware(
       CORSMiddleware,
       allow_origins=origins,
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   logger.info(f"CORS enabled for origins: {origins}")


# API Routers
API_V1_PREFIX = "/api/v1"
app.include_router(router_system.router, prefix=API_V1_PREFIX + "/system", tags=["System & Schema"])
app.include_router(router_openrouter.router, prefix=API_V1_PREFIX + "/openrouter", tags=["OpenRouter Management"])
app.include_router(router_servers.router, prefix=API_V1_PREFIX + "/servers", tags=["External Server Management"])
app.include_router(router_tasks.router, prefix=API_V1_PREFIX + "/tasks", tags=["Task Orchestration & AI Agent"])

# WebSocket Endpoint
# Note: WebSocket routes are typically added directly to the app instance
app.add_websocket_route("/ws/{client_id}", websocket_manager.websocket_endpoint)
logger.info("WebSocket endpoint enabled at /ws/{client_id}")


# Root endpoint for basic health check / welcome
@app.get("/", tags=["Root"])
async def root():
   return {
       "message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}",
       "environment": settings.ENVIRONMENT,
       "documentation": "/docs"
   }

# To make them available via Depends:
app.dependency_overrides[MainAIAgent] = lambda: app_lifespan_globals["main_ai_agent"]
app.dependency_overrides[OpenRouterClient] = lambda: app_lifespan_globals["openrouter_client"]
app.dependency_overrides[ServerRegistry] = lambda: app_lifespan_globals["server_registry"]
app.dependency_overrides[ConnectionManager] = lambda: app_lifespan_globals["connection_manager"]
app.dependency_overrides[TaskLoadBalancer] = lambda: app_lifespan_globals["task_load_balancer"]

logger.info("Application setup complete. Ready to serve requests.")
EOF

   # app/core/__init__.py
   touch "${backend_path}/app/core/__init__.py"

   # app/core/config.py
   cat << EOF > "${backend_path}/app/core/config.py"
#!/usr/bin/env python3
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
   APP_NAME: str = "AI Command Center Backend"
   APP_VERSION: str = "1.1.0"
   DEBUG: bool = True
   ENVIRONMENT: str = "development" # development, staging, production
   SECRET_KEY: str = "your_very_strong_random_secret_key_for_jwt_and_other_crypto"
   ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
   CORS_ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"] # Adjust for Flutter dev port

   # Database (Example: PostgreSQL)
   DB_ENGINE: Optional[str] = None
   DB_USER: Optional[str] = None
   DB_PASSWORD: Optional[str] = None
   DB_HOST: Optional[str] = None
   DB_PORT: Optional[int] = None
   DB_NAME: Optional[str] = None
   DATABASE_URL: Optional[str] = None # Will be constructed if components are set

   # Redis Configuration
   REDIS_HOST: str = "localhost"
   REDIS_PORT: int = 6379
   REDIS_DB_CELERY: int = 0
   REDIS_DB_CACHE: int = 1
   REDIS_PASSWORD: Optional[str] = None

   CELERY_BROKER_URL: str = "redis://localhost:6379/0" # Default, can be overridden by env
   CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0" # Default

   # OpenRouter Configuration
   OPENROUTER_API_KEY: Optional[str] = None
   OPENROUTER_SITE_URL: Optional[str] = "https://youraicommancenter.com"
   OPENROUTER_APP_NAME: Optional[str] = "AICommandCenter"
   OPENROUTER_DEFAULT_MODEL: str = "anthropic/claude-3-sonnet-20240229"
   OPENROUTER_MODEL_CACHE_TTL_SECONDS: int = 3600

   # External Server Communication
   EXTERNAL_SERVER_PIN_SALT: str = "another_strong_random_salt_for_pin_hashing"
   XOR_ENCRYPTION_KEY: str = "a_simple_key_for_xor_example_use_aes_in_prod"

   # Monitoring
   PROMETHEUS_ENABLED: bool = True

   # Token Management & Budgeting
   DEFAULT_TOKEN_BUDGET_USD: float = 100.00
   BUDGET_WARNING_THRESHOLD_PERCENT: int = 80

   # Pydantic-Settings configuration
   model_config = SettingsConfigDict(
       env_file=".env",
       env_file_encoding="utf-8",
       extra="ignore" # Ignore extra fields from .env
   )

   # Post-processing to construct URLs if not set directly
   def __init__(self, **values):
       super().__init__(**values)
       if not self.DATABASE_URL and self.DB_ENGINE and self.DB_USER and self.DB_HOST and self.DB_NAME:
           self.DATABASE_URL = f"{self.DB_ENGINE}://{self.DB_USER}:{self.DB_PASSWORD or ''}@{self.DB_HOST}:{self.DB_PORT or 5432}/{self.DB_NAME}"
       
       redis_auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
       if self.CELERY_BROKER_URL == "redis://localhost:6379/0": # Default value check
            self.CELERY_BROKER_URL = f"redis://{redis_auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_CELERY}"
       if self.CELERY_RESULT_BACKEND == "redis://localhost:6379/0": # Default value check
           self.CELERY_RESULT_BACKEND = f"redis://{redis_auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_CELERY}"


settings = Settings()
EOF

   # app/core/ai_agent.py
   cat << EOF > "${backend_path}/app/core/ai_agent.py"
#!/usr/bin/env python3
import logging
from typing import Dict, List, Any, Optional
from ..models_schemas.requests.task_requests import UserExpectationRequest, TaskDefinition
from ..models_schemas.responses.task_responses import ExecutionPlanResponse, TaskDelegationResult, AgentFeedback
from .openrouter_client import OpenRouterClient
from ..services.orchestration.load_balancer import TaskLoadBalancer
# from ..tasks.celery_tasks import delegate_micro_task_to_executor # Celery task
# from ..utils.token_management import TokenManager # Placeholder
from ..core.config import settings # Import settings here

logger = logging.getLogger(__name__)

class MainAIAgent:
   def __init__(self,
                openrouter_client: OpenRouterClient,
                task_load_balancer: TaskLoadBalancer,
                # token_manager: TokenManager, # TODO: Implement and inject
                # db_session: AsyncSession # If using DB
               ):
       self.openrouter_client = openrouter_client
       self.task_load_balancer = task_load_balancer
       # self.token_manager = token_manager
       self.claude_sonnet_model = settings.OPENROUTER_DEFAULT_MODEL # Configurable

   async def understand_user_expectations(self, user_input: UserExpectationRequest) -> Dict[str, Any]:
       """
       Interacts with Claude Sonnet 4 (via OpenRouter) to precisely understand user expectations.
       """
       logger.info(f"Understanding expectations for user input: {user_input.prompt[:100]}...")
       # TODO: Implement advanced prompt engineering for Claude.
       # Consider context, history, and specific instructions for plan generation.
       prompt_to_claude = f"""
       Analyze the following user request for a mobile application and provide a structured breakdown of their core needs,
       key features, and overall objectives. The output should be a JSON object containing:
       "core_need": "The primary problem the user is trying to solve.",
       "key_features": ["A list of essential features."],
       "objectives": ["High-level goals for the application."],
       "constraints": ["Any constraints mentioned or implied."]

       User Request: "{user_input.prompt}"
       """
       try:
           # TODO: Add token tracking with self.token_manager
           response = await self.openrouter_client.chat_completion(
               model_id=self.claude_sonnet_model,
               messages=[{"role": "user", "content": prompt_to_claude}],
               response_format={"type": "json_object"} # If model supports JSON mode
           )
           # TODO: Parse Claude's JSON response robustly
           # For now, assuming direct JSON content if model supports it, or parse from text
           if response and response.get("choices") and response["choices"][0].get("message"):
               content = response["choices"][0]["message"].get("content")
               # Try to parse content as JSON
               import json
               try:
                   parsed_content = json.loads(content)
                   logger.info("Successfully parsed Claude's response for user expectations.")
                   return parsed_content
               except json.JSONDecodeError:
                   logger.error(f"Failed to parse JSON from Claude's response: {content}")
                   # Fallback or error handling
                   return {"error": "Failed to parse Claude's response", "raw_content": content}
           else:
               logger.error(f"Invalid or empty response from Claude: {response}")
               return {"error": "Invalid response from Claude"}

       except Exception as e:
           logger.error(f"Error interacting with Claude for understanding expectations: {e}", exc_info=True)
           return {"error": str(e)}

   async def develop_execution_plan(self, insights: Dict[str, Any]) -> ExecutionPlanResponse:
       """
       Based on understood expectations, develops a detailed plan (list of tasks).
       This could also involve another call to Claude or a rule-based system.
       """
       logger.info("Developing execution plan based on insights...")
       # TODO: Transform insights into a list of TaskDefinition objects.
       # This might involve another LLM call to break down features into actionable tasks.
       # Example: "key_feature: User Authentication" -> Tasks: "Design auth UI", "Implement backend auth logic", "Test auth flow"
       
       tasks = []
       # Placeholder logic:
       if "key_features" in insights:
           for i, feature in enumerate(insights["key_features"]):
               tasks.append(TaskDefinition(
                   task_id=f"task_{i+1:03d}",
                   name=f"Implement Feature: {feature}",
                   description=f"Detailed steps to implement '{feature}'. This includes UI, backend, testing.",
                   priority=1,
                   estimated_complexity="medium", # small, medium, large
                   dependencies=[] # List of other task_ids
               ))
       
       if not tasks:
            tasks.append(TaskDefinition(
                   task_id="task_fallback_001",
                   name="Initial Project Setup",
                   description="Define overall architecture and initial project scaffolding.",
                   priority=0, estimated_complexity="small", dependencies=[]
               ))

       logger.info(f"Execution plan developed with {len(tasks)} tasks.")
       return ExecutionPlanResponse(plan_id="plan_123", tasks=tasks)

   async def delegate_and_monitor_task(self, task: TaskDefinition) -> TaskDelegationResult:
       """
       Delegates a single, concrete task to an appropriate executor (OpenRouter model or external server)
       selected by the TaskLoadBalancer. This would typically involve Celery.
       """
       logger.info(f"Preparing to delegate task: {task.task_id} - {task.name}")

       # 1. Select executor using Load Balancer
       executor_choice = await self.task_load_balancer.select_executor(task)
       logger.info(f"Task {task.task_id} will be routed to: {executor_choice.executor_type} (ID: {executor_choice.executor_id})")

       # 2. Prepare task data for the executor
       # This is highly dependent on the task and executor type
       # For LLM tasks, it would be a prompt. For code generation, it might be specs.
       task_payload_for_executor = {
           "task_description": task.description,
           "required_output_format": "e.g., Python code snippet, JSON data, text analysis",
           # ... other relevant details from task or project context
       }

       # 3. Delegate using Celery (asynchronous execution)
       # This is a conceptual call. The actual Celery task would handle the API call.
       # celery_task_result = delegate_micro_task_to_executor.delay(
       #     task_id=task.task_id,
       #     executor_type=executor_choice.executor_type,
       #     executor_id=executor_choice.executor_id,
       #     executor_config=executor_choice.executor_config, # API key, URL, etc.
       #     task_payload=task_payload_for_executor
       # )
       # logger.info(f"Task {task.task_id} delegated via Celery. Celery task ID: {celery_task_result.id}")
       
       # For now, simulate direct execution for placeholder purposes
       logger.warning(f"SIMULATING direct execution for task {task.task_id} (Celery not fully implemented in this script).")
       simulated_result = f"Simulated result for {task.task_id} from {executor_choice.executor_type}."
       
       return TaskDelegationResult(
           task_id=task.task_id,
           # celery_task_id=celery_task_result.id,
           celery_task_id=f"simulated_celery_{task.task_id}", # Simulated
           status="PENDING", # Celery task status initially
           message=f"Task delegated to {executor_choice.executor_type}. Waiting for completion.",
           # Store simulated result for immediate feedback in this placeholder
           # In reality, this would come from Celery result backend
           _simulated_result_for_analysis=simulated_result
       )

   async def analyze_feedback_and_iterate(self, task_id: str, execution_result: Any) -> AgentFeedback:
       """
       Analyzes the result from a worker model/server.
       Introduces necessary improvements or approves the work.
       This might involve another LLM call for quality assessment or refinement.
       """
       logger.info(f"Analyzing feedback for task {task_id}. Result: {str(execution_result)[:100]}...")
       
       # TODO: Implement sophisticated analysis logic.
       # This could involve:
       # 1. Validation against expected output format/schema.
       # 2. Quality checks (e.g., code linting, factual correctness for text).
       # 3. Using another LLM (e.g., Claude) to review the output.
       #    Example prompt to Claude: "Review this generated code snippet for quality and adherence to requirements: {execution_result}"
       
       # Placeholder logic:
       is_approved = True # Assume good for now
       comments = "Placeholder: Result looks good."
       refined_result = execution_result # No refinement in this placeholder

       if not is_approved:
           # TODO: Logic for requesting revisions or re-delegating with modifications
           logger.info(f"Task {task_id} requires iteration. Comments: {comments}")
           # refined_result = await self.request_refinement(task_id, execution_result, comments)
       else:
           logger.info(f"Task {task_id} approved.")

       return AgentFeedback(
           task_id=task_id,
           is_approved=is_approved,
           feedback_summary=comments,
           refined_data=refined_result if is_approved else None,
           requires_further_action=not is_approved
       )

   async def save_approved_work(self, task_id: str, approved_data: Any):
       """
       Saves the approved work to a persistent store (e.g., database, file system).
       """
       logger.info(f"Saving approved work for task {task_id}...")
       # TODO: Implement database saving logic.
       # Example:
       # async with self.db_session() as session:
       #     db_record = CompletedTask(task_id=task_id, data=approved_data)
       #     session.add(db_record)
       #     await session.commit()
       logger.info(f"Work for task {task_id} saved (simulated).")
       return {"status": "success", "message": f"Work for {task_id} saved."}
EOF

   # app/core/openrouter_client.py
   cat << EOF > "${backend_path}/app/core/openrouter_client.py"
#!/usr/bin/env python3
import httpx # Using httpx for async requests
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from ..core.config import settings # Assuming settings is globally accessible or passed

logger = logging.getLogger(__name__)

# --- Pydantic Models for OpenRouter API Responses (simplified) ---
class ModelChoice(BaseModel):
   id: str
   name: str
   description: Optional[str] = None
   pricing: Dict[str, float] # E.g., {"prompt": "0.000003", "completion": "0.000006"}
   context_length: Optional[int] = None
   # Add other relevant fields from OpenRouter's /models endpoint

class ModelsResponse(BaseModel):
   data: List[ModelChoice]

class ChatCompletionMessage(BaseModel):
   role: str # "system", "user", "assistant"
   content: str

class ChatCompletionChoice(BaseModel):
   index: int
   message: ChatCompletionMessage
   finish_reason: Optional[str] = None

class ChatCompletionUsage(BaseModel):
   prompt_tokens: int
   completion_tokens: int
   total_tokens: int
   # cost: Optional[float] = None # OpenRouter might add cost in response

class ChatCompletionResponse(BaseModel):
   id: str
   object: str
   created: int
   model: str
   choices: List[ChatCompletionChoice]
   usage: ChatCompletionUsage
   # system_fingerprint: Optional[str] = None # If provided by model

class OpenRouterClient:
   BASE_URL = "https://openrouter.ai/api/v1"

   def __init__(self,
                api_key: str = settings.OPENROUTER_API_KEY,
                site_url: str = settings.OPENROUTER_SITE_URL,
                app_name: str = settings.OPENROUTER_APP_NAME,
                model_cache_ttl: int = settings.OPENROUTER_MODEL_CACHE_TTL_SECONDS):
       if not api_key:
           logger.error("OpenRouter API key is required but not provided.")
           raise ValueError("OpenRouter API key is required.")
       self.api_key = api_key
       self.site_url = site_url
       self.app_name = app_name
       self.model_cache_ttl = model_cache_ttl
       
       self.headers = {
           "Authorization": f"Bearer {self.api_key}",
           "Content-Type": "application/json",
           "HTTP-Referer": self.site_url or "",
           "X-Title": self.app_name or "",
       }
       self._available_models: List[ModelChoice] = []
       self._models_last_fetched_time: float = 0.0
       self._async_client = httpx.AsyncClient(headers=self.headers, timeout=30.0) # Default timeout

   async def close(self):
       """Closes the underlying HTTPX client. Call during application shutdown."""
       await self._async_client.aclose()

   async def get_available_models(self, force_refresh: bool = False) -> List[ModelChoice]:
       """
       Fetches and caches the list of available models from OpenRouter.
       Uses a TTL for caching to avoid excessive API calls.
       """
       current_time = time.time()
       if not force_refresh and self._available_models and \
          (current_time - self._models_last_fetched_time < self.model_cache_ttl):
           logger.debug("Returning cached OpenRouter models.")
           return self._available_models

       logger.info(f"Fetching available models from OpenRouter (Force refresh: {force_refresh})...")
       try:
           response = await self._async_client.get(f"{self.BASE_URL}/models")
           response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
           
           parsed_response = ModelsResponse(**response.json())
           self._available_models = parsed_response.data
           self._models_last_fetched_time = current_time
           logger.info(f"Successfully fetched and cached {len(self._available_models)} models from OpenRouter.")
           return self._available_models
       except httpx.HTTPStatusError as e:
           logger.error(f"HTTP error fetching models from OpenRouter: {e.response.status_code} - {e.response.text}", exc_info=True)
       except httpx.RequestError as e:
           logger.error(f"Request error fetching models from OpenRouter: {e}", exc_info=True)
       except Exception as e: # Includes JSONDecodeError, Pydantic ValidationError
           logger.error(f"Unexpected error processing models response from OpenRouter: {e}", exc_info=True)
       
       # Return stale cache if available on error, otherwise empty list
       return self._available_models if self._available_models else []

   async def refresh_available_models(self):
       """Convenience method to force a refresh of the model list."""
       await self.get_available_models(force_refresh=True)

   async def get_model_details(self, model_id: str) -> Optional[ModelChoice]:
       """Retrieves details for a specific model, using the cache if available."""
       if not self._available_models or time.time() - self._models_last_fetched_time > self.model_cache_ttl:
           await self.refresh_available_models()
       
       for model in self._available_models:
           if model.id == model_id:
               return model
       logger.warning(f"Model ID '{model_id}' not found in available models.")
       return None

   async def chat_completion(self,
                             model_id: str,
                             messages: List[Dict[str, str]], # List of {"role": "...", "content": "..."}
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None,
                             stream: bool = False, # Streaming not fully handled here, returns full response
                             response_format: Optional[Dict[str, str]] = None, # E.g. {"type": "json_object"}
                             **kwargs) -> Optional[Dict[str, Any]]: # Return raw JSON for flexibility
       """
       Performs a chat completion request to OpenRouter.
       """
       logger.debug(f"Requesting chat completion from model '{model_id}'. Messages: {len(messages)}")
       payload = {
           "model": model_id,
           "messages": messages,
           "temperature": temperature,
           **kwargs # Allows passing other valid OpenRouter parameters
       }
       if max_tokens:
           payload["max_tokens"] = max_tokens
       if stream:
           logger.warning("Streaming requested but not fully implemented in this client version; will attempt non-streaming.")
       if response_format:
           payload["response_format"] = response_format

       try:
           response = await self._async_client.post(f"{self.BASE_URL}/chat/completions", json=payload)
           response.raise_for_status()
           completion_data = response.json()
           return completion_data # Return raw JSON
       except httpx.HTTPStatusError as e:
           logger.error(f"HTTP error during chat completion with '{model_id}': {e.response.status_code} - {e.response.text}", exc_info=True)
           return {"error": {"status_code": e.response.status_code, "message": e.response.text, "model_id": model_id}}
       except httpx.RequestError as e:
           logger.error(f"Request error during chat completion with '{model_id}': {e}", exc_info=True)
           return {"error": {"message": str(e), "model_id": model_id}}
       except Exception as e: # Includes JSONDecodeError, Pydantic ValidationError
           logger.error(f"Unexpected error processing chat completion response from '{model_id}': {e}", exc_info=True)
           return {"error": {"message": str(e), "model_id": model_id}}
EOF

   # app/services/discovery/__init__.py
   touch "${backend_path}/app/services/discovery/__init__.py"

   # app/services/discovery/server_registry.py
   cat << EOF > "${backend_path}/app/services/discovery/server_registry.py"
#!/usr/bin/env python3
import logging
import time
from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, HttpUrl, validator, Field # Added Field
from passlib.context import CryptContext # For PIN hashing

from ...core.config import settings # Relative import for settings

logger = logging.getLogger(__name__)

# CryptContext for hashing PINs
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


class ServerCapabilities(BaseModel):
   supported_models: List[str] = []
   task_types: List[str] = [] # e.g., "code_generation", "text_analysis", "image_generation"
   max_concurrent_requests: Optional[int] = None
   custom_metadata: Dict[str, Any] = {}

class RegisteredServer(BaseModel):
   server_id: str
   url: HttpUrl
   name: Optional[str] = None # User-friendly name
   hashed_pin: str # Store hashed PIN, not the plain PIN
   capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
   is_healthy: bool = False
   last_health_check: Optional[float] = None # Timestamp
   latency_ms: Optional[float] = None
   registered_at: float = Field(default_factory=time.time)
   last_updated_at: float = Field(default_factory=time.time)

class ServerRegistry:
   """
   Manages a persistent (or in-memory for this example) registry of external AI servers.
   """
   def __init__(self, storage_backend=None): # storage_backend could be a DB client
       self._servers: Dict[str, RegisteredServer] = {} # In-memory storage
       self.storage_backend = storage_backend
       # TODO: Load servers from persistent storage on init if storage_backend is provided

   def _generate_server_id(self, url: HttpUrl) -> str:
       # Simple ID generation, consider UUIDs for production
       import hashlib
       return f"extsrv_{hashlib.md5(str(url).encode()).hexdigest()[:8]}"

   def _hash_pin(self, pin: str) -> str:
       """Hashes the PIN using bcrypt."""
       return pwd_context.hash(pin)

   def verify_pin(self, pin: str, hashed_pin: str) -> bool:
       """Verifies a plain PIN against a stored hashed PIN."""
       try:
           return pwd_context.verify(pin, hashed_pin)
       except Exception as e: # Catches various errors from passlib
           logger.error(f"Error verifying PIN: {e}")
           return False

   async def register_server(self, url: HttpUrl, pin: str, name: Optional[str] = None) -> RegisteredServer:
       """
       Registers a new external server.
       """
       server_id = self._generate_server_id(url)
       if server_id in self._servers:
           logger.warning(f"Server with URL {url} (ID: {server_id}) already registered. Updating.")
       
       hashed_pin = self._hash_pin(pin)
       server = RegisteredServer(
           server_id=server_id,
           url=url,
           name=name or f"External Server @ {url.host}",
           hashed_pin=hashed_pin
       )
       self._servers[server_id] = server
       logger.info(f"Server '{server.name}' (ID: {server_id}) registered with URL: {url}. PIN stored securely.")
       return server

   async def get_server(self, server_id: str) -> Optional[RegisteredServer]:
       return self._servers.get(server_id)

   async def list_servers(self) -> List[RegisteredServer]:
       return list(self._servers.values())

   async def remove_server(self, server_id: str) -> bool:
       if server_id in self._servers:
           del self._servers[server_id]
           logger.info(f"Server ID {server_id} removed from registry.")
           return True
       logger.warning(f"Attempted to remove non-existent server ID: {server_id}")
       return False

   async def update_server_status(self, server_id: str, is_healthy: bool, latency_ms: Optional[float] = None):
       server = await self.get_server(server_id)
       if server:
           server.is_healthy = is_healthy
           server.latency_ms = latency_ms
           server.last_health_check = time.time()
           server.last_updated_at = time.time()
           logger.debug(f"Server {server_id} status updated: Healthy={is_healthy}, Latency={latency_ms}ms")
       else:
           logger.warning(f"Cannot update status for non-existent server ID: {server_id}")

   async def update_server_capabilities(self, server_id: str, capabilities: ServerCapabilities):
       server = await self.get_server(server_id)
       if server:
           server.capabilities = capabilities
           server.last_updated_at = time.time()
           logger.info(f"Server {server_id} capabilities updated: {capabilities.model_dump_json(indent=2)}")
       else:
           logger.warning(f"Cannot update capabilities for non-existent server ID: {server_id}")

   async def find_servers_by_capability(self, required_model: Optional[str] = None, task_type: Optional[str] = None) -> List[RegisteredServer]:
       matched_servers = []
       for server in self._servers.values():
           if not server.is_healthy:
               continue
           
           match = True
           if required_model and required_model not in server.capabilities.supported_models:
               match = False
           if task_type and task_type not in server.capabilities.task_types:
               match = False
           
           if match:
               matched_servers.append(server)
       return matched_servers
EOF

   # app/services/discovery/connection_manager.py
   cat << EOF > "${backend_path}/app/services/discovery/connection_manager.py"
#!/usr/bin/env python3
import asyncio
import httpx
import logging
import time
from typing import Optional, Dict, Any
from .server_registry import ServerRegistry, ServerCapabilities, RegisteredServer
from ...security.encryption import xor_encrypt_decrypt # Example XOR encryption
from ...core.config import settings

logger = logging.getLogger(__name__)

class ConnectionManager:
   """
   Manages continuous health monitoring, latency testing, and capability discovery
   for registered external servers.
   """
   def __init__(self, server_registry: ServerRegistry, health_check_interval_seconds: int = 60):
       self.server_registry = server_registry
       self.health_check_interval = health_check_interval_seconds
       self._monitoring_task: Optional[asyncio.Task] = None
       self._async_client = httpx.AsyncClient(timeout=10.0) # Timeout for health checks

   async def _perform_health_check(self, server: RegisteredServer) -> tuple[bool, Optional[float]]:
       """Performs a health check and measures latency."""
       health_check_url = f"{str(server.url).rstrip('/')}/health" # Standard health endpoint
       start_time = time.perf_counter()
       try:
           response = await self._async_client.get(health_check_url)
           latency_ms = (time.perf_counter() - start_time) * 1000
           
           if response.status_code == 200:
               logger.debug(f"Health check for {server.server_id} ({server.url}) successful. Latency: {latency_ms:.2f}ms")
               return True, latency_ms
           else:
               logger.warning(f"Health check for {server.server_id} ({server.url}) failed. Status: {response.status_code}")
               return False, latency_ms
       except httpx.TimeoutException:
           logger.warning(f"Health check for {server.server_id} ({server.url}) timed out.")
           return False, None
       except httpx.RequestError as e:
           logger.warning(f"Health check for {server.server_id} ({server.url}) request error: {e}")
           return False, None
       except Exception as e:
           logger.error(f"Unexpected error during health check for {server.server_id}: {e}", exc_info=True)
           return False, None

   async def _discover_capabilities(self, server: RegisteredServer, pin: str) -> Optional[ServerCapabilities]:
       """
       Queries a server for its capabilities.
       """
       capabilities_url = f"{str(server.url).rstrip('/')}/capabilities"
       request_payload = {"timestamp": time.time()}
       
       encrypted_payload_str = xor_encrypt_decrypt(str(request_payload), settings.XOR_ENCRYPTION_KEY)
       
       headers = {
           "Content-Type": "application/octet-stream",
           "X-PIN": pin
       }
       
       logger.debug(f"Attempting to discover capabilities for {server.server_id} at {capabilities_url}")
       try:
           response = await self._async_client.post(capabilities_url, content=encrypted_payload_str.encode(), headers=headers)
           
           if response.status_code == 200:
               decrypted_response_str = xor_encrypt_decrypt(response.text, settings.XOR_ENCRYPTION_KEY)
               import json
               try:
                   capabilities_data = json.loads(decrypted_response_str)
                   capabilities = ServerCapabilities(**capabilities_data)
                   logger.info(f"Successfully discovered capabilities for {server.server_id}: {capabilities.model_dump_json(indent=2)}")
                   return capabilities
               except (json.JSONDecodeError, TypeError) as e:
                   logger.error(f"Error parsing capabilities JSON from {server.server_id}: {e}. Response: {decrypted_response_str[:200]}")
                   return None
           else:
               logger.warning(f"Failed to discover capabilities for {server.server_id}. Status: {response.status_code}, Response: {response.text[:200]}")
               return None
       except httpx.RequestError as e:
           logger.warning(f"Request error during capability discovery for {server.server_id}: {e}")
           return None
       except Exception as e:
           logger.error(f"Unexpected error during capability discovery for {server.server_id}: {e}", exc_info=True)
           return None

   async def check_and_update_server(self, server: RegisteredServer, provided_pin_for_capabilities: Optional[str] = None):
       """
       Performs health check and, if healthy and PIN provided, capability discovery.
       """
       is_healthy, latency_ms = await self._perform_health_check(server)
       await self.server_registry.update_server_status(server.server_id, is_healthy, latency_ms)

       if is_healthy and provided_pin_for_capabilities:
           capabilities = await self._discover_capabilities(server, provided_pin_for_capabilities)
           if capabilities:
               await self.server_registry.update_server_capabilities(server.server_id, capabilities)
       elif is_healthy and not provided_pin_for_capabilities:
            logger.debug(f"Server {server.server_id} is healthy, but no PIN provided for capability discovery during this check.")


   async def _monitor_servers_loop(self):
       """Periodically checks all registered servers."""
       logger.info("Starting server monitoring loop...")
       while True:
           try:
               servers_to_check = await self.server_registry.list_servers()
               if not servers_to_check:
                   logger.debug("No servers registered to monitor.")
               else:
                   logger.info(f"Performing periodic health checks for {len(servers_to_check)} servers...")
               
               check_tasks = [
                   self.check_and_update_server(server)
                   for server in servers_to_check
               ]
               await asyncio.gather(*check_tasks, return_exceptions=True)

           except asyncio.CancelledError:
               logger.info("Server monitoring loop cancelled.")
               break
           except Exception as e:
               logger.error(f"Error in server monitoring loop: {e}", exc_info=True)
           
           await asyncio.sleep(self.health_check_interval)

   async def start_monitoring(self):
       """Starts the background server monitoring task."""
       if self._monitoring_task is None or self._monitoring_task.done():
           self._monitoring_task = asyncio.create_task(self._monitor_servers_loop())
           logger.info("Server monitoring background task started.")
       else:
           logger.warning("Server monitoring task already running.")

   async def stop_monitoring(self):
       """Stops the background server monitoring task."""
       if self._monitoring_task and not self._monitoring_task.done():
           self._monitoring_task.cancel()
           try:
               await self._monitoring_task
           except asyncio.CancelledError:
               logger.info("Server monitoring background task successfully stopped.")
       self._monitoring_task = None
       await self._async_client.aclose()
EOF
   
   # app/services/orchestration/__init__.py
   touch "${backend_path}/app/services/orchestration/__init__.py"

   # app/services/orchestration/load_balancer.py
   cat << EOF > "${backend_path}/app/services/orchestration/load_balancer.py"
#!/usr/bin/env python3
import logging
import random
from typing import List, Optional, NamedTuple
from ...core.openrouter_client import OpenRouterClient, ModelChoice
from ..discovery.server_registry import ServerRegistry, RegisteredServer
from ...models_schemas.requests.task_requests import TaskDefinition
from ...core.config import settings

logger = logging.getLogger(__name__)

class ExecutorChoice(NamedTuple):
   executor_type: str # "openrouter" or "external_server"
   executor_id: str   # OpenRouter model ID or external server_id
   executor_config: Optional[dict] = None # e.g., API key for OpenRouter, URL for server
   reason: str # Why this executor was chosen

class TaskLoadBalancer:
   """
   Intelligent routing of tasks.
   Considers server load, capabilities, response times, cost, performance history.
   """
   def __init__(self, server_registry: ServerRegistry, openrouter_client: OpenRouterClient):
       self.server_registry = server_registry
       self.openrouter_client = openrouter_client

   async def _get_candidate_openrouter_models(self, task: TaskDefinition) -> List[ModelChoice]:
       """Finds suitable OpenRouter models based on task requirements."""
       all_models = await self.openrouter_client.get_available_models()
       candidate_models = [
           model for model in all_models 
           if (model.context_length or 8000) > (task.estimated_input_tokens or 1000)
       ]
       return candidate_models or all_models

   async def _get_candidate_external_servers(self, task: TaskDefinition) -> List[RegisteredServer]:
       """Finds suitable external servers based on task requirements."""
       all_servers = await self.server_registry.list_servers()
       healthy_servers = [s for s in all_servers if s.is_healthy]
       return healthy_servers

   async def _score_executor(self, executor, task: TaskDefinition, executor_type: str) -> float:
       """
       Scores a potential executor. Higher score is better.
       """
       score = 0.0
       
       if executor_type == "openrouter":
           model: ModelChoice = executor
           avg_price = (float(model.pricing.get("prompt", 1.0)) + float(model.pricing.get("completion", 1.0))) / 2
           score += (1 / (avg_price * 1_000_000)) if avg_price > 0 else 1.0

       elif executor_type == "external_server":
           server: RegisteredServer = executor
           if not server.is_healthy: return -float('inf')

           if server.latency_ms is not None:
               score += 100 / (server.latency_ms + 1)

       score += random.uniform(0, 0.1)
       return score

   async def select_executor(self, task: TaskDefinition) -> ExecutorChoice:
       """
       Selects the best executor for the task.
       """
       logger.info(f"Selecting executor for task: {task.task_id} - {task.name}")
       candidate_choices: List[Tuple[float, ExecutorChoice]] = []

       # Consider OpenRouter models
       openrouter_models = await self._get_candidate_openrouter_models(task)
       for model in openrouter_models:
           score = await self._score_executor(model, task, "openrouter")
           if score > -float('inf'):
               candidate_choices.append((score, ExecutorChoice(
                   executor_type="openrouter",
                   executor_id=model.id,
                   executor_config={"api_key": settings.OPENROUTER_API_KEY},
                   reason=f"OpenRouter model. Score: {score:.2f}"
               )))

       # Consider External Servers
       external_servers = await self._get_candidate_external_servers(task)
       for server in external_servers:
           score = await self._score_executor(server, task, "external_server")
           if score > -float('inf'):
               candidate_choices.append((score, ExecutorChoice(
                   executor_type="external_server",
                   executor_id=server.server_id,
                   executor_config={"url": str(server.url), "hashed_pin": server.hashed_pin},
                   reason=f"External server. Score: {score:.2f}, Healthy: {server.is_healthy}, Latency: {server.latency_ms}"
               )))
       
       if not candidate_choices:
           logger.error(f"No suitable executor found for task {task.task_id}.")
           default_model_id = settings.OPENROUTER_DEFAULT_MODEL
           return ExecutorChoice(
               executor_type="openrouter",
               executor_id=default_model_id,
               executor_config={"api_key": settings.OPENROUTER_API_KEY},
               reason=f"Fallback: No suitable executor found, using default OpenRouter model {default_model_id}."
           )

       candidate_choices.sort(key=lambda x: x[0], reverse=True)
       best_score, best_choice = candidate_choices[0]
       
       logger.info(f"Best executor for task {task.task_id}: {best_choice.executor_type} - {best_choice.executor_id} (Score: {best_score:.2f})")
       logger.debug(f"Reason for choice: {best_choice.reason}")
       if len(candidate_choices) > 1:
           logger.debug(f"Other candidates considered: {[(c[1].executor_id, c[0]) for c in candidate_choices[1:4]]}")

       return best_choice
EOF

   # app/tasks/__init__.py
   touch "${backend_path}/app/tasks/__init__.py"

   # app/tasks/celery_app.py
   cat << EOF > "${backend_path}/app/tasks/celery_app.py"
#!/usr/bin/env python3
from celery import Celery
from ...core.config import settings # Relative import from tasks to core

celery_app = Celery(
   "ai_command_center_worker",
   broker=settings.CELERY_BROKER_URL,
   backend=settings.CELERY_RESULT_BACKEND,
   include=[
       'app.tasks.celery_tasks'
   ]
)

celery_app.conf.update(
   task_serializer='json',
   result_serializer='json',
   accept_content=['json'],
   timezone='UTC',
   enable_utc=True,
)

if __name__ == '__main__':
   celery_app.start()
EOF

   # app/tasks/celery_tasks.py
   cat << EOF > "${backend_path}/app/tasks/celery_tasks.py"
#!/usr/bin/env python3
import logging
import time
import httpx 
from .celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="delegate_micro_task_to_executor")
def delegate_micro_task_to_executor(self, task_id: str, executor_type: str, executor_id: str, executor_config: dict, task_payload: dict):
   """
   This Celery task is responsible for making the actual API call to the selected executor.
   'bind=True' makes 'self' (the task instance) available.
   """
   logger.info(f"Celery task started for original task_id: {task_id} -> executor: {executor_type}/{executor_id}")
   self.update_state(state='PROGRESS', meta={'task_id': task_id, 'status': 'Executing API call'})

   # Placeholder logic for making the API call
   try:
       # Example for OpenRouter
       if executor_type == "openrouter":
           # In a real app, you might re-instantiate a client or use a global one carefully.
           # For simplicity, we'll make a direct httpx call.
           api_key = executor_config.get("api_key")
           headers = {"Authorization": f"Bearer {api_key}"}
           # The actual payload for OpenRouter needs to be constructed from task_payload
           # This is a major simplification.
           or_payload = {
               "model": executor_id,
               "messages": [{"role": "user", "content": task_payload.get("task_description", "No description")}]
           }
           with httpx.Client() as client:
               response = client.post("https://openrouter.ai/api/v1/chat/completions", json=or_payload, headers=headers, timeout=120)
               response.raise_for_status()
               result = response.json()

       # Example for External Server
       elif executor_type == "external_server":
           server_url = executor_config.get("url")
           # The external server might have its own API spec
           with httpx.Client() as client:
               response = client.post(f"{server_url}/execute_task", json=task_payload, timeout=120)
               response.raise_for_status()
               result = response.json()
       
       else:
           raise ValueError(f"Unknown executor type: {executor_type}")

       logger.info(f"Celery task for {task_id} completed successfully.")
       # The return value of the task will be stored in the result backend
       return {"task_id": task_id, "status": "SUCCESS", "result": result}

   except httpx.HTTPStatusError as e:
       logger.error(f"HTTP error in Celery task {task_id}: {e.response.status_code} - {e.response.text}")
       # Mark the task as failed
       self.update_state(state='FAILURE', meta={'task_id': task_id, 'exc_type': 'HTTPStatusError', 'exc_message': str(e)})
       # You can re-raise the exception to have Celery mark it as FAILED
       raise
   except Exception as e:
       logger.error(f"Generic error in Celery task {task_id}: {e}", exc_info=True)
       self.update_state(state='FAILURE', meta={'task_id': task_id, 'exc_type': str(type(e).__name__), 'exc_message': str(e)})
       raise
EOF

   # celery_worker_runner.py
   cat << EOF > "${backend_path}/celery_worker_runner.py"
#!/usr/bin/env python3
# This file is used to run the Celery worker.
# Ensure your PYTHONPATH includes the project root or run from the backend directory.
# Command: celery -A app.tasks.celery_app worker -l info -P gevent (or eventlet for async tasks with httpx)
# Or for solo pool (simpler, good for dev/debug): celery -A app.tasks.celery_app worker -l info --pool=solo

from app.tasks.celery_app import celery_app

if __name__ == '__main__':
   print("This script is a placeholder. To run the Celery worker, use the command:")
   print("  celery -A app.tasks.celery_app worker -l info --pool=solo")
   print("Ensure you are in the 'backend' directory or your PYTHONPATH is set correctly.")
EOF
   log_message "Generated celery_worker_runner.py."

   # --- API Routers ---
   
   # app/api/v1/__init__.py
   touch "${backend_path}/app/api/v1/__init__.py"

   # app/api/v1/router_openrouter.py
   cat << EOF > "${backend_path}/app/api/v1/router_openrouter.py"
#!/usr/bin/env python3
from fastapi import APIRouter, HTTPException, Body, Depends, Query, Request
from typing import List, Optional, Any
from ...core.openrouter_client import OpenRouterClient
from ...models_schemas.requests.openrouter_requests import APIKeyConfigRequest, ChatCompletionRequest
from ...models_schemas.responses.openrouter_responses import ModelChoiceResponse, ModelListResponse, ChatCompletionApiResponse
from ...core.config import settings

router = APIRouter()

async def get_or_client(request: Request) -> OpenRouterClient:
   client = getattr(request.app.state, "openrouter_client", None)
   if not client:
       raise HTTPException(status_code=503, detail="OpenRouter client not available.")
   return client

@router.post("/configure/api_key", summary="Configure or Update OpenRouter API Key", response_model=dict)
async def configure_api_key_endpoint(
   payload: APIKeyConfigRequest,
   or_client: OpenRouterClient = Depends(get_or_client)
):
   if not payload.api_key:
       raise HTTPException(status_code=400, detail="API key cannot be empty.")
   
   or_client.api_key = payload.api_key
   or_client.headers["Authorization"] = f"Bearer {payload.api_key}"
   await or_client.refresh_available_models()
   return {"message": "OpenRouter API key updated. Model list is being refreshed."}

@router.get("/models", response_model=ModelListResponse, summary="List Available OpenRouter Models")
async def list_openrouter_models_endpoint(
   or_client: OpenRouterClient = Depends(get_or_client),
   force_refresh: bool = Query(False, description="Force refresh of model list from OpenRouter API")
) -> ModelListResponse:
   models_data = await or_client.get_available_models(force_refresh=force_refresh)
   if not models_data:
       raise HTTPException(status_code=503, detail="Could not fetch models from OpenRouter.")
   
   response_models = [ModelChoiceResponse.model_validate(model.model_dump()) for model in models_data]
   return ModelListResponse(models=response_models)

@router.get("/models/{model_id}", response_model=ModelChoiceResponse, summary="Get Details for a Specific Model")
async def get_openrouter_model_details_endpoint(
   model_id: str,
   or_client: OpenRouterClient = Depends(get_or_client)
) -> ModelChoiceResponse:
   model_detail_data = await or_client.get_model_details(model_id)
   if not model_detail_data:
       raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")
   return ModelChoiceResponse.model_validate(model_detail_data.model_dump())

@router.post("/chat/completions", response_model=ChatCompletionApiResponse, summary="Proxy Chat Completions to OpenRouter")
async def create_openrouter_chat_completion_endpoint(
   request_payload: ChatCompletionRequest,
   or_client: OpenRouterClient = Depends(get_or_client)
) -> ChatCompletionApiResponse:
   messages_for_client = [msg.model_dump() for msg in request_payload.messages]
   raw_response = await or_client.chat_completion(
       model_id=request_payload.model,
       messages=messages_for_client,
       temperature=request_payload.temperature,
       max_tokens=request_payload.max_tokens,
       stream=request_payload.stream,
       response_format=request_payload.response_format.model_dump() if request_payload.response_format else None
   )
   if not raw_response or "error" in raw_response:
       error_info = raw_response.get("error", {}) if raw_response else {}
       status_code = error_info.get("status_code", 500)
       raise HTTPException(status_code=status_code, detail=f"OpenRouter API error: {error_info.get('message', 'Unknown error')}")
   try:
       return ChatCompletionApiResponse.model_validate(raw_response)
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error parsing OpenRouter response: {e}")
EOF
   log_message "Generated app/api/v1/router_openrouter.py."

   # app/api/v1/router_servers.py
   cat << EOF > "${backend_path}/app/api/v1/router_servers.py"
#!/usr/bin/env python3
import asyncio
import time
from fastapi import APIRouter, HTTPException, Body, Depends, Path, Query, Request
from typing import List, Optional
from pydantic import HttpUrl

from ...services.discovery.server_registry import ServerRegistry
from ...services.discovery.connection_manager import ConnectionManager
from ...models_schemas.requests.server_requests import ServerRegistrationRequest, ServerUpdateRequest, ServerPinRequest
from ...models_schemas.responses.server_responses import ServerResponse, ServerListResponse, ServerStatusResponse, ServerCapabilitiesResponse
from ...models_schemas.common import MessageResponse

router = APIRouter()

async def get_server_registry(request: Request) -> ServerRegistry:
   registry = getattr(request.app.state, "server_registry", None)
   if not registry: raise HTTPException(status_code=503, detail="Server Registry not available.")
   return registry

async def get_connection_manager(request: Request) -> ConnectionManager:
   manager = getattr(request.app.state, "connection_manager", None)
   if not manager: raise HTTPException(status_code=503, detail="Connection Manager not available.")
   return manager

@router.post("/register", response_model=ServerResponse, status_code=201, summary="Register a New External Server")
async def register_external_server_endpoint(
   payload: ServerRegistrationRequest,
   registry: ServerRegistry = Depends(get_server_registry),
   conn_manager: ConnectionManager = Depends(get_connection_manager)
):
   try:
       registered_server_data = await registry.register_server(
           url=payload.url, pin=payload.pin, name=payload.name
       )
       asyncio.create_task(conn_manager.check_and_update_server(registered_server_data, payload.pin))
       return ServerResponse.model_validate(registered_server_data.model_dump())
   except ValueError as e:
       raise HTTPException(status_code=400, detail=str(e))
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to register server: {e}")

@router.get("/", response_model=ServerListResponse, summary="List All Registered External Servers")
async def list_external_servers_endpoint(
   registry: ServerRegistry = Depends(get_server_registry)
) -> ServerListResponse:
   servers_data = await registry.list_servers()
   response_servers = [ServerResponse.model_validate(s.model_dump()) for s in servers_data]
   return ServerListResponse(servers=response_servers)

@router.get("/{server_id}", response_model=ServerResponse, summary="Get Details of a Specific Server")
async def get_server_details_endpoint(
   server_id: str = Path(..., description="The unique ID of the server"),
   registry: ServerRegistry = Depends(get_server_registry)
) -> ServerResponse:
   server_data = await registry.get_server(server_id)
   if not server_data:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   return ServerResponse.model_validate(server_data.model_dump())

@router.put("/{server_id}", response_model=ServerResponse, summary="Update an Existing Server's Information")
async def update_server_info_endpoint(
   server_id: str = Path(..., description="The ID of the server to update"),
   payload: ServerUpdateRequest = Body(...),
   registry: ServerRegistry = Depends(get_server_registry),
   conn_manager: ConnectionManager = Depends(get_connection_manager)
):
   existing_server = await registry.get_server(server_id)
   if not existing_server:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   update_data = payload.model_dump(exclude_unset=True)
   for key, value in update_data.items():
       if hasattr(existing_server, key) and value is not None:
           setattr(existing_server, key, value)
   existing_server.last_updated_at = time.time()
   if 'url' in update_data:
        asyncio.create_task(conn_manager.check_and_update_server(existing_server))
   return ServerResponse.model_validate(existing_server.model_dump())

@router.post("/{server_id}/update_pin", response_model=MessageResponse, summary="Update a Server's PIN")
async def update_server_pin_endpoint(
   server_id: str = Path(..., description="The ID of the server whose PIN to update"),
   payload: ServerPinRequest = Body(...),
   registry: ServerRegistry = Depends(get_server_registry)
):
   existing_server = await registry.get_server(server_id)
   if not existing_server:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   new_hashed_pin = registry._hash_pin(payload.new_pin)
   existing_server.hashed_pin = new_hashed_pin
   existing_server.last_updated_at = time.time()
   return MessageResponse(message="Server PIN updated successfully.")

@router.delete("/{server_id}", response_model=MessageResponse, summary="Deregister an External Server")
async def deregister_server_endpoint(
   server_id: str = Path(..., description="The ID of the server to deregister"),
   registry: ServerRegistry = Depends(get_server_registry)
):
   success = await registry.remove_server(server_id)
   if not success:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   return MessageResponse(message=f"Server '{server_id}' deregistered successfully.")

@router.post("/{server_id}/health_check", response_model=ServerStatusResponse, summary="Trigger Manual Health Check")
async def trigger_server_health_check_endpoint(
   server_id: str = Path(..., description="The ID of the server to health check"),
   registry: ServerRegistry = Depends(get_server_registry),
   conn_manager: ConnectionManager = Depends(get_connection_manager)
):
   server = await registry.get_server(server_id)
   if not server:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   await conn_manager.check_and_update_server(server)
   updated_server = await registry.get_server(server_id)
   return ServerStatusResponse(
       server_id=updated_server.server_id,
       is_healthy=updated_server.is_healthy,
       latency_ms=updated_server.latency_ms,
       last_health_check=updated_server.last_health_check
   )

@router.post("/{server_id}/discover_capabilities", response_model=ServerCapabilitiesResponse, summary="Trigger Manual Capability Discovery")
async def trigger_server_capability_discovery_endpoint(
   server_id: str = Path(..., description="The ID of the server for capability discovery"),
   payload: ServerPinRequest = Body(..., description="Server's current PIN for authentication"),
   registry: ServerRegistry = Depends(get_server_registry),
   conn_manager: ConnectionManager = Depends(get_connection_manager)
):
   server = await registry.get_server(server_id)
   if not server:
       raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found.")
   if not registry.verify_pin(payload.new_pin, server.hashed_pin):
       raise HTTPException(status_code=403, detail="Incorrect PIN provided.")
   capabilities = await conn_manager._discover_capabilities(server, payload.new_pin)
   if capabilities:
       await registry.update_server_capabilities(server.server_id, capabilities)
       return ServerCapabilitiesResponse.model_validate(capabilities.model_dump())
   else:
       raise HTTPException(status_code=500, detail="Failed to discover capabilities.")
EOF
   log_message "Generated app/api/v1/router_servers.py."

   # app/api/v1/router_tasks.py
   cat << EOF > "${backend_path}/app/api/v1/router_tasks.py"
#!/usr/bin/env python3
from fastapi import APIRouter, Body, Depends, Path, HTTPException, Request
from typing import List, Any
from ...core.ai_agent import MainAIAgent
from ...models_schemas.requests.task_requests import UserExpectationRequest, TaskDefinition, TaskExecutionFeedbackRequest
from ...models_schemas.responses.task_responses import AgentUnderstandingResponse, ExecutionPlanResponse, TaskDelegationResult, AgentFeedbackResponse, TaskStatusResponse
from ...tasks.celery_app import celery_app
from celery.result import AsyncResult

router = APIRouter()

async def get_main_ai_agent(request: Request) -> MainAIAgent:
   agent = getattr(request.app.state, "main_ai_agent", None)
   if not agent: raise HTTPException(status_code=503, detail="Main AI Agent not available.")
   return agent

@router.post("/understand_expectations", response_model=AgentUnderstandingResponse, summary="AI Understands User Expectations")
async def understand_user_expectations_endpoint(
   payload: UserExpectationRequest,
   agent: MainAIAgent = Depends(get_main_ai_agent)
):
   understanding_result = await agent.understand_user_expectations(payload)
   if "error" in understanding_result:
       raise HTTPException(status_code=500, detail=f"Error understanding expectations: {understanding_result['error']}")
   return AgentUnderstandingResponse(
       user_prompt=payload.prompt,
       understood_needs=understanding_result 
   )

@router.post("/develop_plan", response_model=ExecutionPlanResponse, summary="AI Develops Execution Plan")
async def develop_execution_plan_endpoint(
   insights: dict = Body(..., example={"core_need": "...", "key_features": [], "objectives": []}),
   agent: MainAIAgent = Depends(get_main_ai_agent)
):
   plan = await agent.develop_execution_plan(insights)
   if not plan.tasks:
       raise HTTPException(status_code=500, detail="Failed to develop a valid execution plan.")
   return plan

@router.post("/delegate_task", response_model=TaskDelegationResult, summary="AI Delegates a Specific Task")
async def delegate_single_task_endpoint(
   task_definition: TaskDefinition,
   agent: MainAIAgent = Depends(get_main_ai_agent)
):
   delegation_info = await agent.delegate_and_monitor_task(task_definition)
   return delegation_info

@router.post("/analyze_task_result", response_model=AgentFeedbackResponse, summary="AI Analyzes Task Result and Iterates")
async def analyze_task_result_endpoint(
   feedback_request: TaskExecutionFeedbackRequest,
   agent: MainAIAgent = Depends(get_main_ai_agent)
):
   feedback = await agent.analyze_feedback_and_iterate(
       task_id=feedback_request.task_id,
       execution_result=feedback_request.execution_result
   )
   return AgentFeedbackResponse.model_validate(feedback.model_dump())

@router.post("/save_approved_work", summary="Save Approved Work to Database", response_model=dict)
async def save_approved_work_endpoint(
   task_id: str = Body(..., embed=True),
   approved_data: Any = Body(..., embed=True),
   agent: MainAIAgent = Depends(get_main_ai_agent)
):
   result = await agent.save_approved_work(task_id, approved_data)
   if result.get("status") != "success":
       raise HTTPException(status_code=500, detail=f"Failed to save work: {result.get('message')}")
   return result

@router.get("/celery_task_status/{celery_task_id}", response_model=TaskStatusResponse, summary="Get Status of a Celery Task")
async def get_celery_task_status_endpoint(
   celery_task_id: str = Path(..., description="The ID of the Celery task")
):
   task_result = AsyncResult(celery_task_id, app=celery_app)
   response_data = {
       "celery_task_id": celery_task_id,
       "status": task_result.status,
       "result": task_result.result if task_result.ready() else None,
       "traceback": task_result.traceback if task_result.failed() else None,
   }
   if task_result.ready() and isinstance(task_result.result, dict) and "task_id" in task_result.result:
       response_data["original_task_id"] = task_result.result["task_id"]
   return TaskStatusResponse.model_validate(response_data)
EOF
   log_message "Generated app/api/v1/router_tasks.py."

   # app/api/v1/router_system.py
   cat << EOF > "${backend_path}/app/api/v1/router_system.py"
#!/usr/bin/env python3
from fastapi import APIRouter, Request, Body
from typing import Dict, Any
from ...core.config import settings
from datetime import datetime, timezone

router = APIRouter()

@router.get("/health", summary="System Health Check", response_model=dict)
async def system_health_check_endpoint(request: Request):
   openrouter_client_status = "Not Initialized"
   client = getattr(request.app.state, "openrouter_client", None)
   if client:
       openrouter_client_status = "Available" if client.api_key else "API Key Missing"
   
   return {
       "status": "healthy",
       "application_name": settings.APP_NAME,
       "version": settings.APP_VERSION,
       "environment": settings.ENVIRONMENT,
       "services": {
           "openrouter_client": openrouter_client_status,
       }
   }

@router.post("/schema/evolve", summary="Dynamically Evolve Communication Schema (Placeholder)", response_model=dict)
async def evolve_communication_schema_endpoint(
   new_schema_definition: Dict[str, Any] = Body(...)
):
   return {
       "message": "Schema evolution request received (placeholder).",
       "new_schema_preview": new_schema_definition
   }
EOF
   log_message "Generated app/api/v1/router_system.py."

   # --- Pydantic Models (models_schemas) ---
   mkdir -p "${backend_path}/app/models_schemas/requests"
   mkdir -p "${backend_path}/app/models_schemas/responses"
   mkdir -p "${backend_path}/app/models_schemas/db"
   touch "${backend_path}/app/models_schemas/__init__.py"
   
   cat << EOF > "${backend_path}/app/models_schemas/common.py"
#!/usr/bin/env python3
from pydantic import BaseModel
from typing import Optional, Any
class MessageResponse(BaseModel):
   message: str
   detail: Optional[Any] = None
EOF
   
   cat << EOF > "${backend_path}/app/models_schemas/requests/openrouter_requests.py"
#!/usr/bin/env python3
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
class APIKeyConfigRequest(BaseModel):
   api_key: str = Field(..., description="The OpenRouter API key.")
class ChatMessage(BaseModel):
   role: str = Field(..., description="Role of the message sender.")
   content: str = Field(..., description="Content of the message.")
class ResponseFormat(BaseModel):
   type: str = Field("text", description="Type of response format.")
class ChatCompletionRequest(BaseModel):
   model: str = Field(..., description="ID of the OpenRouter model.")
   messages: List[ChatMessage]
   temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
   max_tokens: Optional[int] = Field(None, gt=0)
   stream: Optional[bool] = Field(False)
   response_format: Optional[ResponseFormat] = None
EOF
   log_message "Ensured Pydantic model schema directories and basic files exist."
   
   # --- Security Layer ---
   mkdir -p "${backend_path}/app/security"
   touch "${backend_path}/app/security/__init__.py"
   
   cat << EOF > "${backend_path}/app/security/encryption.py"
#!/usr/bin/env python3
# Placeholder for encryption utilities (XOR, AES)
def xor_encrypt_decrypt(data: str, key: str) -> str:
   # Simple XOR cipher for demonstration. Not for production use.
   if not data:
       return ""
   return "".join(chr(ord(c) ^ ord(k)) for c, k in zip(data, key * (len(data) // len(key) + 1)))

class AESCipher: # Placeholder
   def __init__(self, key: bytes): pass
   def encrypt(self, plaintext: str) -> str: return plaintext
   def decrypt(self, ciphertext: str) -> Optional[str]: return ciphertext
EOF
   log_message "Ensured security directory and basic files exist."

   # --- Utilities ---
   mkdir -p "${backend_path}/app/utils"
   touch "${backend_path}/app/utils/__init__.py"
   cat << EOF > "${backend_path}/app/utils/token_management.py"
#!/usr/bin/env python3
# Placeholder for TokenManager class
class TokenManager: pass # Placeholder
EOF
   log_message "Ensured utils directory and basic files exist."

   # --- WebSockets ---
   mkdir -p "${backend_path}/app/websockets"
   touch "${backend_path}/app/websockets/__init__.py"
   cat << EOF > "${backend_path}/app/websockets/manager.py"
#!/usr/bin/env python3
# Placeholder for WebSocketConnectionManager
class WebSocketConnectionManager:
   async def connect(self, websocket, client_id, user_info=None): await websocket.accept()
   def disconnect(self, client_id): pass
   async def handle_incoming_message(self, client_id, raw_message): pass
   async def websocket_endpoint(self, websocket, client_id):
     await self.connect(websocket, client_id)
     try:
         while True:
             data = await websocket.receive_text()
             await self.handle_incoming_message(client_id, data)
     except Exception: pass
     finally: self.disconnect(client_id)

websocket_manager = WebSocketConnectionManager()
EOF
   log_message "Ensured websockets directory and basic files exist."

   # --- Monitoring ---
   mkdir -p "${backend_path}/app/monitoring"
   touch "${backend_path}/app/monitoring/__init__.py"
   cat << EOF > "${backend_path}/app/monitoring/prometheus.py"
#!/usr/bin/env python3
# Placeholder for PrometheusMonitor
from starlette.responses import PlainTextResponse
class PrometheusMonitor:
   async def handle_metrics(self, request):
       return PlainTextResponse("")
EOF
   log_message "Ensured monitoring directory and basic files exist."
   
   # --- Tests Directory ---
   mkdir -p "${backend_path}/tests/api"
   mkdir -p "${backend_path}/tests/services"
   mkdir -p "${backend_path}/tests/tasks"
   touch "${backend_path}/tests/__init__.py"
   
   cat << EOF > "${backend_path}/tests/conftest.py"
import pytest
from httpx import AsyncClient
from fastapi import FastAPI

@pytest.fixture(scope="session")
def app() -> FastAPI:
   from app.main import app as main_app
   return main_app

@pytest.fixture
async def async_client(app: FastAPI) -> AsyncClient:
   async with AsyncClient(app=app, base_url="http://testserver") as client:
       yield client
EOF
   log_message "Generated tests directory structure and conftest.py."
   
   cat << EOF > "${backend_path}/README.md"
# AI Command Center - Backend (Advanced)
This is the enterprise-grade backend for the AI Command Center, built with FastAPI and Celery.
This version includes more detailed skeletons for advanced features.
## Features
- Core AI Agent System with OpenRouter Integration (Claude Sonnet focus)
- External Server Discovery, Health Monitoring, and Capability Management
- Intelligent Task Orchestration (Rule-based load balancing placeholder, Celery for async tasks)
- WebSocket support for real-time communication.
- Placeholder for enterprise features like advanced security, monitoring, and structured logging.
## API Documentation
Available at \`/docs\` (Swagger UI) and \`/redoc\` (ReDoc) when the server is running.
The API is versioned under \`/api/v1\`.
EOF
   log_message "Ensured comprehensive backend README.md is in place."
   
   log_message "DETAILED Backend structure and files created."
}

create_frontend_files() {
   log_message "Creating DETAILED frontend structure and files (Flutter - Cyberpunk Command Center)..."
   local frontend_path="${BUILD_DIR}/${PROJECT_NAME}/frontend"
   mkdir -p "${frontend_path}/lib/core/theme" 
   mkdir -p "${frontend_path}/lib/core/config"
   mkdir -p "${frontend_path}/lib/core/constants"
   mkdir -p "${frontend_path}/lib/core/navigation"
   mkdir -p "${frontend_path}/lib/models/openrouter"
   mkdir -p "${frontend_path}/lib/models/server"
   mkdir -p "${frontend_path}/lib/models/task"
   mkdir -p "${frontend_path}/lib/providers" 
   mkdir -p "${frontend_path}/lib/services/api" 
   mkdir -p "${frontend_path}/lib/services/websocket"
   mkdir -p "${frontend_path}/lib/widgets/common"
   mkdir -p "${frontend_path}/lib/widgets/effects" 
   mkdir -p "${frontend_path}/lib/widgets/navigation"
   mkdir -p "${frontend_path}/lib/features/dashboard/screens"
   mkdir -p "${frontend_path}/lib/features/dashboard/widgets"
   mkdir -p "${frontend_path}/lib/features/openrouter_config/screens"
   mkdir -p "${frontend_path}/lib/features/openrouter_config/widgets"
   mkdir -p "${frontend_path}/lib/features/server_management/screens"
   mkdir -p "${frontend_path}/lib/features/server_management/widgets"
   mkdir -p "${frontend_path}/lib/features/task_orchestration/screens"
   mkdir -p "${frontend_path}/lib/features/task_orchestration/widgets"
   mkdir -p "${frontend_path}/lib/features/settings/screens"
   mkdir -p "${frontend_path}/lib/utils"
   mkdir -p "${frontend_path}/assets/fonts"
   mkdir -p "${frontend_path}/assets/images/textures"
   mkdir -p "${frontend_path}/assets/images/logos"
   mkdir -p "${frontend_path}/assets/icons_svg"
   mkdir -p "${frontend_path}/assets/shaders"
   # README.md for frontend (updated)
   cat << EOF > "${frontend_path}/README.md"
# AI Command Center - Flutter Frontend (Advanced)

This is the premium Flutter frontend for the AI Command Center, featuring a Cyberpunk theme.
This version includes more detailed skeletons for UI components and features.

## Features
- **Deep Space Cyberpunk Theme**: Immersive visual design with neon accents, glassmorphism, and planned advanced animations.
- **Component Skeletons**:
 - OpenRouter Configuration Panel (API Key, Model List, Usage Analytics)
 - Server Management Interface (Server List, Add/Details Dialogs)
 - Central Command Center Dashboard (Live Activity, System Metrics, Cost Tracking, Quick Actions)
 - Visual Flow Designer (Placeholder for canvas, node palette, properties)
 - Settings Screen
- **State Management**: Riverpod example setup.
- **Navigation**: GoRouter example setup with persistent navigation shell.
- **Responsive Design Considerations**: Placeholders for adapting to different screen sizes.

## Setup
1.  Ensure you have the Flutter SDK installed and configured (see [Flutter official documentation](https://flutter.dev/docs/get-started/install)).
   It's recommended to use a Flutter version manager like [FVM](https://fvm.app/).
2.  Navigate to this \`frontend\` directory: \`cd frontend\`
3.  Get dependencies: \`flutter pub get\`
4.  Run code generation (if using packages like Riverpod generator, GoRouter generator with build_runner):
   \`flutter pub run build_runner build --delete-conflicting-outputs\`
5.  Run the app: \`flutter run\`
   - Select a device or emulator when prompted.
   - For web, you might need to specify the Chrome device: \`flutter run -d chrome\`
   - Pass environment variables for API URLs if configured (e.g., using \`--dart-define\`).

## Building for Production
- **Android APK**: \`flutter build apk --release\`
 (Output: \`build/app/outputs/flutter-apk/app-release.apk\`)
- **Android App Bundle**: \`flutter build appbundle --release\`
 (Output: \`build/app/outputs/bundle/release/app-release.aab\`)
- **iOS**: Requires macOS and Xcode. \`flutter build ipa\`
- **Web**: \`flutter build web --release\`
 (Output: \`build/web\`)
 Ensure your web server is configured correctly for a Flutter web app (e.g., handling routing).

## Project Structure Highlights
- \`lib/core\`: Theme, config, navigation, constants.
- \`lib/features\`: Main application features, each with its own \`screens\` and \`widgets\` subdirectories.
- \`lib/models\`: Data models for API responses and application state.
- \`lib/providers\`: State management logic (Riverpod providers).
- \`lib/services\`: API client services, WebSocket service.
- \`lib/widgets/common\`: Reusable UI components (e.g., GlassmorphismPanel, NeonButton).
- \`assets\`: Fonts, images, icons, shaders.

## Development Notes
- **State Management**: This project uses Riverpod as an example. Adapt to your preferred solution (Provider, BLoC, GetX, etc.) if needed.
- **Navigation**: GoRouter is used for declarative routing.
- **Styling**: Adhere to the \`CyberpunkTheme\` for consistent UI.
- **Code Generation**: If you modify files that require code generation (e.g., Riverpod providers with \`@riverpod\`, GoRouter routes if using a generator), remember to run the build_runner command.
- **TODOs**: Look for \`// TODO:\` comments in the code for areas requiring further implementation or attention.
EOF
   log_message "DETAILED Frontend structure and files created."
}

create_docs_files() {
   log_message "Creating DETAILED documentation structure and files..."
   local docs_path="${BUILD_DIR}/${PROJECT_NAME}/docs"
   mkdir -p "${docs_path}/C4_diagrams"
   mkdir -p "${docs_path}/adr" # Architecture Decision Records
   mkdir -p "${docs_path}/user_guide"
   mkdir -p "${docs_path}/developer_guide"

   cat << EOF > "${docs_path}/README.md"
# AI Command Center - Project Documentation Hub

This directory serves as the central repository for all documentation related to the AI Command Center project.
Effective documentation is crucial for understanding, developing, maintaining, and using this enterprise-grade system.

## Table of Contents

### 1. System Overview & Architecture
-   [**System Vision & Goals**](./system_vision.md): High-level purpose and objectives of the AI Command Center.
-   [**Functional Requirements**](./functional_requirements.md): What the system does.
-   [**Non-Functional Requirements**](./non_functional_requirements.md): Qualities like performance, security, scalability.
-   [**System Architecture (High-Level)**](./developer_guide/architecture_overview.md): Key components and their interactions.
-   [**C4 Model Diagrams**](./C4_diagrams/README.md): Visual architecture (Context, Containers, Components).
   -   [System Context Diagram](./C4_diagrams/level1_system_context.md)
   -   [Container Diagram](./C4_diagrams/level2_container.md)
   -   [Component Diagrams](./C4_diagrams/level3_component.md)
-   [**Technology Stack**](./developer_guide/technology_stack.md): Chosen technologies and rationale.
-   [**Data Model Overview**](./developer_guide/data_model.md): Key data entities and relationships (conceptual).

### 2. Developer Guides
-   [**Developer Onboarding**](./developer_guide/onboarding.md): Steps for new developers.
-   [**Development Environment Setup**](./developer_guide/setup_guide.md): Detailed setup for backend and frontend.
-   [**Backend Development Guide**](./developer_guide/backend_development.md): Conventions, patterns, key modules.
-   [**Frontend Development Guide**](./developer_guide/frontend_development.md): Conventions, state management, UI components.
-   [**API Specification (OpenAPI)**](./developer_guide/api_specification.md): Details of the backend API (link to live docs).
-   [**Coding Standards & Best Practices**](./developer_guide/coding_standards.md): Guidelines for code quality.
-   [**Testing Strategy**](./developer_guide/testing_strategy.md): Unit, integration, E2E tests.
-   [**Debugging Guide**](./developer_guide/debugging_guide.md): Tips for troubleshooting.
-   [**Contribution Guidelines**](./developer_guide/CONTRIBUTING.md): How to contribute to the project.

### 3. User & Operations Guides
-   [**User Manual**](./user_guide/README.md): How to use the AI Command Center application.
   -   [Getting Started for Users](./user_guide/getting_started.md)
   -   [Dashboard Overview](./user_guide/dashboard.md)
   -   [OpenRouter Configuration](./user_guide/openrouter_config.md)
   -   [Managing External Servers](./user_guide/external_servers.md)
   -   [Task Orchestration & Visual Designer](./user_guide/task_orchestration.md)
-   [**Deployment Guide**](./ops_guide/deployment.md): Instructions for deploying the system.
-   [**Monitoring & Maintenance**](./ops_guide/monitoring_maintenance.md): How to monitor the system and perform routine maintenance.
-   [**Security Overview & Procedures**](./ops_guide/security_procedures.md): Key security aspects and operational procedures.
-   [**Troubleshooting Common Issues (Ops)**](./ops_guide/troubleshooting.md)

### 4. Project Management & Design
-   [**Roadmap**](./project_management/roadmap.md): Future plans and features.
-   [**Architecture Decision Records (ADRs)**](./adr/README.md): Log of significant architectural decisions.
   -   [ADR-001: Choice of FastAPI for Backend](./adr/ADR-001_FastAPI.md)
   -   [ADR-002: Choice of Flutter for Frontend](./adr/ADR-002_Flutter.md)
-   [**UI/UX Design Philosophy**](./design/ui_ux_philosophy.md): Principles behind the Cyberpunk Command Center design.
-   [**Glossary of Terms**](./glossary.md): Definitions of key terms used in the project.

---
*This documentation structure aims to be comprehensive for an enterprise-grade project.*
*Fill in the content for each Markdown file as the project progresses.*
EOF

   # Create placeholder files for the new docs structure
   touch "${docs_path}/system_vision.md"
   touch "${docs_path}/functional_requirements.md"
   touch "${docs_path}/non_functional_requirements.md"
   touch "${docs_path}/glossary.md"

   mkdir -p "${docs_path}/developer_guide"
   touch "${docs_path}/developer_guide/architecture_overview.md"
   touch "${docs_path}/developer_guide/technology_stack.md"
   touch "${docs_path}/developer_guide/data_model.md"
   touch "${docs_path}/developer_guide/onboarding.md"
   mv "${docs_path}/setup_guide.md" "${docs_path}/developer_guide/setup_guide.md" 2>/dev/null || touch "${docs_path}/developer_guide/setup_guide.md"
   touch "${docs_path}/developer_guide/backend_development.md"
   touch "${docs_path}/developer_guide/frontend_development.md"
   mv "${docs_path}/api_specification.md" "${docs_path}/developer_guide/api_specification.md" 2>/dev/null || touch "${docs_path}/developer_guide/api_specification.md"
   touch "${docs_path}/developer_guide/coding_standards.md"
   touch "${docs_path}/developer_guide/testing_strategy.md"
   touch "${docs_path}/developer_guide/debugging_guide.md"
   touch "${docs_path}/developer_guide/CONTRIBUTING.md"

   mkdir -p "${docs_path}/user_guide"
   touch "${docs_path}/user_guide/README.md"
   touch "${docs_path}/user_guide/getting_started.md"
   touch "${docs_path}/user_guide/dashboard.md"
   touch "${docs_path}/user_guide/openrouter_config.md"
   touch "${docs_path}/user_guide/external_servers.md"
   touch "${docs_path}/user_guide/task_orchestration.md"

   mkdir -p "${docs_path}/ops_guide"
   mv "${docs_path}/deployment.md" "${docs_path}/ops_guide/deployment.md" 2>/dev/null || touch "${docs_path}/ops_guide/deployment.md"
   touch "${docs_path}/ops_guide/monitoring_maintenance.md"
   touch "${docs_path}/ops_guide/security_procedures.md"
   touch "${docs_path}/ops_guide/troubleshooting.md"

   mkdir -p "${docs_path}/project_management"
   touch "${docs_path}/project_management/roadmap.md"

   mkdir -p "${docs_path}/design"
   touch "${docs_path}/design/ui_ux_philosophy.md"

   # ADR Files
   cat << EOF > "${docs_path}/adr/README.md"
# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the AI Command Center project.
ADRs document important architectural decisions, their context, and consequences.
We use a format similar to Michael Nygard's ADRs.

## ADR Log
-   [ADR-001: Choice of FastAPI for Backend](./ADR-001_FastAPI.md)
-   [ADR-002: Choice of Flutter for Frontend](./ADR-002_Flutter.md)
-   [ADR-003: State Management for Flutter (Riverpod)](./ADR-003_Flutter_State_Riverpod.md)
-   [ADR-004: Navigation for Flutter (GoRouter)](./ADR-004_Flutter_Navigation_GoRouter.md)
-   [ADR-005: Asynchronous Task Processing (Celery with Redis)](./ADR-005_Celery_Redis.md)
-   *(Add more as decisions are made)*
EOF
   cat << EOF > "${docs_path}/adr/ADR-001_FastAPI.md"
# ADR-001: Choice of FastAPI for Backend Framework

-   **Status**: Accepted
-   **Date**: $(date +%Y-%m-%d)
-   **Deciders**: Project Lead / AI Assistant

## Context and Problem Statement
We need a robust, high-performance, and modern Python framework for building the backend API of the AI Command Center. Key requirements include asynchronous support (for I/O-bound operations like LLM calls), automatic data validation and serialization, OpenAPI documentation generation, and ease of development.

## Considered Options
1.  **FastAPI**: Modern, high-performance web framework based on Starlette and Pydantic.
2.  **Django REST framework (DRF)**: Mature and widely used, built on Django.
3.  **Flask (+ extensions)**: Lightweight and flexible, requires more manual setup for features like OpenAPI.

## Decision Outcome
Chosen option: **FastAPI**.

### Rationale
-   **Performance**: FastAPI is one of the fastest Python frameworks available, thanks to Starlette and Pydantic.
-   **Asynchronous Support**: Native \`async/await\` support is crucial for handling concurrent LLM API calls and WebSocket connections efficiently.
-   **Data Validation & Serialization**: Pydantic integration provides automatic request/response validation and serialization, reducing boilerplate and errors.
-   **Automatic API Docs**: Built-in OpenAPI and JSON Schema generation simplifies API documentation and client generation.
-   **Developer Experience**: Modern Python type hints, dependency injection system, and clear structure lead to a good developer experience.
-   **Ecosystem**: Growing ecosystem and good community support.
-   **Suitability for AI/ML**: Well-suited for ML model serving and integrating with asynchronous libraries.

DRF is powerful but can be heavier, and its ORM-centric nature is not strictly required for all microservices. Flask is flexible but requires more effort to achieve the same level of out-of-the-box features as FastAPI.

## Consequences
-   Requires Python 3.7+ (this project targets 3.9+).
-   Team needs to be familiar with Pydantic and asynchronous programming in Python.
-   The ecosystem, while growing, might be less mature in some specific areas compared to Django's vast plugin library (though FastAPI is often self-sufficient).
EOF
   # Add more ADR stubs
   touch "${docs_path}/adr/ADR-002_Flutter.md"
   touch "${docs_path}/adr/ADR-003_Flutter_State_Riverpod.md"
   touch "${docs_path}/adr/ADR-004_Flutter_Navigation_GoRouter.md"
   touch "${docs_path}/adr/ADR-005_Celery_Redis.md"

   # C4 Diagrams
   cat << EOF > "${docs_path}/C4_diagrams/README.md"
# C4 Model Diagrams for AI Command Center

This section contains diagrams illustrating the architecture of the AI Command Center using the C4 model (Context, Containers, Components, Code). These diagrams help in understanding the system at different levels of abstraction.

For generating these diagrams, tools like Structurizr, PlantUML, or Diagrams.net (with C4 stencils) are recommended.

## Diagram Index
1.  [**Level 1: System Context Diagram**](./level1_system_context.md)
   -   Shows the AI Command Center system as a single black box, its users, and its interactions with key external systems (OpenRouter, External AI Servers).
2.  [**Level 2: Container Diagram**](./level2_container.md)
   -   Zooms into the AI Command Center system, showing its major deployable/runnable units (containers) like the Frontend App, Backend API, Celery Workers, Database, etc.
3.  [**Level 3: Component Diagrams**](./level3_component.md)
   -   Zooms into specific containers to show their internal components and interactions.
   -   Example: Backend API Components (Routers, Services, AI Agent Core, DB Access).
   -   Example: Frontend App Components (Feature Modules, UI Widgets, State Management, API Client).
4.  **Level 4: Code Diagrams (Optional)**
   -   Detailed diagrams (e.g., UML class diagrams) for specific, complex components. Usually created on-demand.

*(The .md files linked above would contain embedded images or PlantUML/Mermaid code for the diagrams.)*
EOF
   touch "${docs_path}/C4_diagrams/level1_system_context.md"
   touch "${docs_path}/C4_diagrams/level2_container.md"
   touch "${docs_path}/C4_diagrams/level3_component.md"

   log_message "DETAILED Documentation structure and files created."
}

create_scripts_files() {
   log_message "Creating utility scripts (build, run, deploy)..."
   local scripts_path="${BUILD_DIR}/${PROJECT_NAME}/scripts"
   mkdir -p "${scripts_path}"

   cat << EOF > "${scripts_path}/build_docker.sh"
#!/bin/bash
set -e

echo "Building AI Command Center Backend Docker image..."
# Assuming this script is in project_root/scripts
cd "$(dirname "$0")/../backend" || { echo "Error: backend directory not found."; exit 1; }


DOCKER_IMAGE_NAME="ai-command-center-backend"
DOCKER_IMAGE_TAG="${SCRIPT_VERSION:-latest}" # Use script version or latest

echo "Using Docker image: \${DOCKER_IMAGE_NAME}:\${DOCKER_IMAGE_TAG}"

docker build -t "\${DOCKER_IMAGE_NAME}:\${DOCKER_IMAGE_TAG}" .

echo ""
echo "Docker image \${DOCKER_IMAGE_NAME}:\${DOCKER_IMAGE_TAG} built successfully."
echo "To run (example):"
echo "  docker run -d -p 8000:8000 \\"
echo "    --env-file .env \\"
echo "    --name aicc-backend \\"
echo "    \${DOCKER_IMAGE_NAME}:\${DOCKER_IMAGE_TAG}"
echo ""
echo "Ensure your .env file is correctly populated in the backend directory."
EOF
   chmod +x "${scripts_path}/build_docker.sh"

   # run_dev.sh
   cat << EOF > "${scripts_path}/run_dev.sh"
#!/bin/bash
# AI Command Center - Development Environment Runner
# This script helps start the backend services (FastAPI, Celery) and provides guidance for Flutter.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
BACKEND_DIR="../backend"
FRONTEND_DIR="../frontend"

FASTAPI_HOST="0.0.0.0"
FASTAPI_PORT="8000"
CELERY_LOG_LEVEL="INFO"
CELERY_CONCURRENCY_POOL="solo" # Default: good for debugging, simple tasks

# --- Helper Functions ---
cleanup() {
   echo ""
   echo "Attempting to stop development services..."
   if [ -n "\$FASTAPI_PID" ]; then kill \$FASTAPI_PID 2>/dev/null && echo "FastAPI server stopped."; fi
   if [ -n "\$CELERY_PID" ]; then kill \$CELERY_PID 2>/dev/null && echo "Celery worker stopped."; fi
   echo "Cleanup complete."
   exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run cleanup
trap cleanup SIGINT SIGTERM

# --- Main Logic ---
echo "🚀 Starting AI Command Center Development Environment..."
echo "----------------------------------------------------"

# 1. Check for Redis (Celery dependency)
if ! redis-cli ping > /dev/null 2>&1; then
   echo "⚠️ WARNING: Redis server does not seem to be running or accessible."
   echo "Celery worker requires Redis. Please start Redis and re-run, or the worker might fail."
else
   echo "✅ Redis server responded to PING."
fi
echo "----------------------------------------------------"

# 2. Start Backend FastAPI Server
echo "🔥 Starting FastAPI backend server on http://\${FASTAPI_HOST}:\${FASTAPI_PORT}..."
echo "Logs will be shown below. Press Ctrl+C to stop all services."
(cd "\$BACKEND_DIR" && uvicorn app.main:app --host "\$FASTAPI_HOST" --port "\$FASTAPI_PORT" --reload) &
FASTAPI_PID=\$!
sleep 2 

if ! kill -0 \$FASTAPI_PID 2>/dev/null; then
   echo "❌ ERROR: FastAPI server failed to start. Check logs above."
   exit 1
fi
echo "✅ FastAPI backend server process started (PID: \$FASTAPI_PID)."
echo "   API Docs: http://localhost:\${FASTAPI_PORT}/docs"
echo "----------------------------------------------------"

# 3. Start Celery Worker
echo "🛠️ Starting Celery worker (Pool: \${CELERY_CONCURRENCY_POOL}, Log Level: \${CELERY_LOG_LEVEL})..."
(cd "\$BACKEND_DIR" && celery -A app.tasks.celery_app worker -l "\$CELERY_LOG_LEVEL" --pool="\$CELERY_CONCURRENCY_POOL") &
CELERY_PID=\$!
sleep 2

if ! kill -0 \$CELERY_PID 2>/dev/null; then
   echo "❌ ERROR: Celery worker failed to start. Check logs above and ensure Redis is running."
   cleanup
   exit 1
fi
echo "✅ Celery worker process started (PID: \$CELERY_PID)."
echo "----------------------------------------------------"

# 4. Frontend Instructions
echo "🎨 For Flutter Frontend (Cyberpunk Command Center):"
echo "   1. Navigate to the frontend directory: cd \"\$FRONTEND_DIR\""
echo "   2. Ensure dependencies are up-to-date: flutter pub get"
echo "   3. If using code generation (Riverpod/GoRouter generators):"
echo "      flutter pub run build_runner build --delete-conflicting-outputs"
echo "   4. Run the app on your chosen device/emulator/web:"
echo "      flutter run"
echo "      (e.g., flutter run -d chrome)"
echo "----------------------------------------------------"

echo "✅ All backend development services initiated."
echo "Monitoring logs. Press Ctrl+C to stop."

wait -n \$FASTAPI_PID \$CELERY_PID
echo "A background service has exited. Initiating cleanup..."
cleanup
EOF
   chmod +x "${scripts_path}/run_dev.sh"

   # deploy_prod.sh
   cat << EOF > "${scripts_path}/deploy_prod.sh"
#!/bin/bash
# AI Command Center - Production Deployment Script (Placeholder)
# This script outlines conceptual steps and needs to be heavily customized
# for your specific production environment (e.g., Kubernetes, Docker Swarm, VMs, Serverless).

set -eo pipefail

# --- Configuration ---
readonly APP_VERSION="\$(git describe --tags --always --dirty || echo 'local-dev')"
readonly DOCKER_REGISTRY="your-docker-registry.com"
readonly BACKEND_IMAGE_NAME="ai-command-center-backend"
# ... more config variables ...

log() { echo "[INFO] \$(date '+%Y-%m-%d %H:%M:%S') - \$1"; }
error() { echo "[ERROR] \$(date '+%Y-%m-%d %H:%M:%S') - \$1" >&2; }
confirm() { read -p "\$1 (y/N): " choice; [[ "\$choice" =~ ^[Yy]$ ]]; }

# --- Deployment Steps (as functions) ---

pre_flight_checks() {
   log "Running pre-flight checks..."
   # ... check for git, docker, flutter, etc. ...
   log "Pre-flight checks passed."
}

build_backend() {
   log "Building backend Docker image: \${DOCKER_REGISTRY}/\${BACKEND_IMAGE_NAME}:\${APP_VERSION}"
   (cd "$(dirname "$0")/../backend" && \
    docker build -t "\${DOCKER_REGISTRY}/\${BACKEND_IMAGE_NAME}:\${APP_VERSION}" .
   ) || { error "Backend Docker build failed."; exit 1; }
   log "Backend Docker image built successfully."
}

# ... more functions for push, build_frontend, deploy_backend etc. ...

# --- Main Deployment Orchestration ---
main_deploy() {
   log "Starting AI Command Center Production Deployment (Version: \${APP_VERSION})"
   
   if ! confirm "Proceed with deployment to PRODUCTION?"; then
       log "Deployment aborted by user."
       exit 0
   fi

   pre_flight_checks
   build_backend
   # ... call other functions ...

   log "🎉 AI Command Center Deployment to Production completed (placeholders)."
}

# --- Script Execution ---
if [ "\$1" == "build_backend" ]; then build_backend;
elif [ -z "\$1" ]; then main_deploy;
else
   echo "Usage: \$0 [build_backend|all]"
   exit 1
fi

exit 0
EOF
   chmod +x "${scripts_path}/deploy_prod.sh"

   # code_quality.sh
   cat << EOF > "${scripts_path}/code_quality.sh"
#!/bin/bash
# AI Command Center - Code Quality Script
# Runs linters and formatters for backend and frontend.

set -e
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="\$(dirname "\$SCRIPT_DIR")"
BACKEND_DIR="\${PROJECT_ROOT}/backend"
FRONTEND_DIR="\${PROJECT_ROOT}/frontend"

echo "🚀 Running Code Quality Checks for AI Command Center..."
echo "----------------------------------------------------"

# --- Backend (Python) ---
echo "🐍 Checking Backend (Python)..."
cd "\$BACKEND_DIR" || exit 1

echo "   Formatting with Black..."
black .

echo "   Linting with Ruff..."
ruff check .

echo "✅ Backend checks complete."
echo "----------------------------------------------------"

# --- Frontend (Flutter/Dart) ---
echo "🎨 Checking Frontend (Flutter/Dart)..."
cd "\$FRONTEND_DIR" || exit 1

echo "   Formatting with 'dart format'..."
dart format .

echo "   Analyzing with 'flutter analyze'..."
flutter analyze

echo "✅ Frontend checks complete."
echo "----------------------------------------------------"
echo "🎉 All code quality checks passed successfully!"
EOF
   chmod +x "${scripts_path}/code_quality.sh"

   log_message "Utility scripts created."
}

create_vscode_config() {
   log_message "Creating VS Code configuration files (launch.json, settings.json, extensions.json)..."
   local vscode_path="${BUILD_DIR}/${PROJECT_NAME}/.vscode"
   mkdir -p "${vscode_path}"

   # launch.json
   cat << EOF > "${vscode_path}/launch.json"
{
   "version": "0.2.0",
   "configurations": [
       // --- Flutter Frontend Configurations ---
       {
           "name": "Flutter: Run All (Frontend - Dev)",
           "cwd": "\${workspaceFolder}/frontend",
           "request": "launch",
           "type": "dart",
           "flutterMode": "debug"
       },
       // --- Python Backend Configurations ---
       {
           "name": "Python: FastAPI (Backend - Dev)",
           "type": "python",
           "request": "launch",
           "module": "uvicorn",
           "args": [
               "app.main:app",
               "--host", "0.0.0.0",
               "--port", "8000",
               "--reload"
           ],
           "cwd": "\${workspaceFolder}/backend",
           "envFile": "\${workspaceFolder}/backend/.env",
           "console": "integratedTerminal"
       },
       {
           "name": "Python: Celery Worker (Backend - Dev)",
           "type": "python",
           "request": "launch",
           "module": "celery",
           "args": [
               "-A", "app.tasks.celery_app",
               "worker",
               "-l", "INFO",
               "--pool=solo"
           ],
           "cwd": "\${workspaceFolder}/backend",
           "envFile": "\${workspaceFolder}/backend/.env",
           "console": "integratedTerminal"
       }
   ],
   "compounds": [
       {
           "name": "Full Stack: FastAPI + Flutter (Dev)",
           "configurations": [
               "Python: FastAPI (Backend - Dev)",
               "Flutter: Run All (Frontend - Dev)"
           ],
           "stopAll": true
       }
   ]
}
EOF

   # settings.json
   cat << EOF > "${vscode_path}/settings.json"
{
   "editor.formatOnSave": true,
   "files.eol": "\\n",
   "files.insertFinalNewline": true,
   "files.trimTrailingWhitespace": true,

   "[python]": {
       "editor.defaultFormatter": "ms-python.black-formatter",
       "editor.tabSize": 4,
       "editor.insertSpaces": true
   },
   "python.testing.pytestArgs": ["backend/tests"],
   "python.testing.unittestEnabled": false,
   "python.testing.pytestEnabled": true,
   
   "[dart]": {
       "editor.defaultFormatter": "Dart-Code.dart-code",
       "editor.tabSize": 2,
       "editor.insertSpaces": true,
       "editor.rulers": [80, 100]
   },
   "dart.lineLength": 100,

   "files.exclude": {
       "**/.git": true,
       "**/.svn": true,
       "**/.hg": true,
       "**/CVS": true,
       "**/.DS_Store": true,
       "**/Thumbs.db": true,
       "backend/__pycache__/": true,
       "backend/.pytest_cache/": true,
       "backend/venv/": true,
       "backend/.venv/": true,
       "frontend/.dart_tool/": true,
       "frontend/build/": true
   },
   "search.exclude": {
       "**/*.g.dart": true,
       "**/*.freezed.dart": true
   }
}
EOF

   # extensions.json
   cat << EOF > "${vscode_path}/extensions.json"
{
   "recommendations": [
       // --- General ---
       "eamodio.gitlens",
       "gruntfuggly.todo-tree",
       "streetsidesoftware.code-spell-checker",
       "ms-azuretools.vscode-docker",
       "github.copilot",

       // --- Python (Backend) ---
       "ms-python.python",
       "ms-python.vscode-pylance",
       "ms-python.black-formatter",
       "charliermarsh.ruff",
       "tamasfe.even-better-toml",
       "redhat.vscode-yaml",
       "humao.rest-client",

       // --- Flutter & Dart (Frontend) ---
       "Dart-Code.dart-code",
       "Dart-Code.flutter",
       "NashAppProductions.flutter-widget-snippets",

       // --- Markdown & Documentation ---
       "yzhang.markdown-all-in-one",
       "davidanson.vscode-markdownlint"
   ]
}
EOF
   log_message "VS Code configuration files created."
}

create_config_files() {
   log_message "Creating root configuration files..."
   local config_path="${BUILD_DIR}/${PROJECT_NAME}/config"
   mkdir -p "${config_path}"

   cat << EOF > "${config_path}/README.md"
# Configuration Files

This directory is intended to hold environment-specific configurations
(e.g., for development, staging, production), if not managed solely by .env files.
EOF

   cat << EOF > "${config_path}/development.json"
{
   "environment": "development",
   "api_base_url": "http://localhost:8000/api/v1",
   "log_level": "DEBUG"
}
EOF
   cat << EOF > "${config_path}/production.json"
{
   "environment": "production",
   "api_base_url": "https://aicommandcenter.yourdomain.com/api/v1",
   "log_level": "WARNING"
}
EOF
   log_message "Root configuration files created."
}

create_root_level_files() {
   log_message "Creating root level files (.gitignore, README.md, pyproject.toml)..."
   local project_root_path="${BUILD_DIR}/${PROJECT_NAME}"
local backend_path="${project_root_path}/backend"

   # .gitignore
   curl -fLo "${project_root_path}/.gitignore" "https://www.toptal.com/developers/gitignore/api/python,django,fastapi,celery,flutter,dart,visualstudiocode,macos,linux,windows" \
   || {
       error_message "Failed to download .gitignore. Creating a basic one."
       cat << EOF > "${project_root_path}/.gitignore"
# Python
__pycache__/
*.py[cod]
.env
.venv/
venv/
dist/
build/

# Flutter
.dart_tool/
.flutter-plugins
.flutter-plugins-dependencies
.packages
.pub-cache/
frontend/build/
ios/Pods/
android/.gradle/

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
EOF
   }
   echo "/${BUILD_DIR_BASE}*/" >> "${project_root_path}/.gitignore"
   echo "/${ZIP_FILENAME}" >> "${project_root_path}/.gitignore"

   # Main README.md
   cat << EOF > "${project_root_path}/README.md"
# AI Command Center - Enterprise AI Management System (Advanced Blueprint)

**Version:** ${SCRIPT_VERSION}

Welcome to the AI Command Center, a flagship, enterprise-grade system designed for sophisticated AI task management,
deep integration with OpenRouter, and intelligent orchestration of work across external AI model execution servers.
This repository contains an advanced blueprint and highly detailed skeleton generated to kickstart development.

## Core Project Vision
The AI Command Center aims to be a central hub for:
-   **Seamless OpenRouter Integration**: Dynamically discover, configure, and utilize a wide array of models from OpenRouter.ai.
-   **Plug-and-Play External Server Management**: Register, monitor, and manage custom AI model execution servers.
-   **Intelligent Task Orchestration**: Employ rule-based or ML-based routing to delegate tasks optimally.
-   **Cyberpunk Command Center UI**: A premium, futuristic, and highly functional Flutter-based user interface.

## Getting Started (Development)

### Prerequisites
-   Python 3.9+
-   Flutter SDK
-   Docker & Docker Compose (Recommended)
-   Redis

### Setup Steps
1.  **Clone/Unzip**: Unzip \`${ZIP_FILENAME}\` and navigate into the project directory: \`cd ${PROJECT_NAME}\`
2.  **Backend Setup (\`backend\` directory)**: See \`backend/README.md\`.
3.  **Frontend Setup (\`frontend\` directory)**: See \`frontend/README.md\`.
4.  **Run Development Servers**: Use the VS Code launch configurations or the script: \`./scripts/run_dev.sh\`.

## Documentation Hub
For comprehensive documentation, including architecture, API specs, and setup guides, please refer to the [**docs**](./docs/README.md) directory.

---
*This project structure and advanced blueprint were generated by the AI Command Center Assembly Script v${SCRIPT_VERSION}.*
EOF

   # pyproject.toml for backend
   cat << EOF > "${backend_path}/pyproject.toml"
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ai_command_center_backend"
version = "${SCRIPT_VERSION}"
description = "Backend for the AI Command Center."
readme = "README.md"
requires-python = ">=3.9"

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "W", "F", "I", "C90", "N", "B", "Q", "SIM", "UP", "RUFF"]
ignore = ["E501", "B008"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101"]

[tool.ruff.lint.isort]
known-first-party = ["app"]

[tool.pytest.ini_options]
pythonpath = ["."]
minversion = "6.0"
addopts = "-ra -q --cov=app --cov-report=html"
testpaths = ["tests"]
EOF

   log_message "Root level files created."
}

package_project() {
   log_message "Creating ZIP archive: ${ZIP_FILENAME}..."
   cd "${BUILD_DIR}" || { error_message "Failed to change to build directory ${BUILD_DIR}"; exit 1; }
   if zip -r "../${ZIP_FILENAME}" "${PROJECT_NAME}"; then
       log_message "Successfully created ZIP archive: ../${ZIP_FILENAME}"
       local abs_zip_path
       if command -v readlink &> /dev/null && readlink -f "../${ZIP_FILENAME}" &> /dev/null; then
           abs_zip_path=$(readlink -f "../${ZIP_FILENAME}")
       else
           abs_zip_path="$(cd "$(dirname "../${ZIP_FILENAME}")" && pwd)/$(basename "../${ZIP_FILENAME}")"
       fi
       log_message "Archive location: ${abs_zip_path}"
   else
       error_message "Failed to create ZIP archive."
       cd ..
       exit 1
   fi
   cd ..
}

cleanup() {
   log_message "Cleaning up build directory: ${BUILD_DIR}..."
   if [ -d "${BUILD_DIR}" ]; then
       if rm -rf "${BUILD_DIR}"; then
           log_message "Build directory cleaned up successfully."
       else
           error_message "Failed to clean up build directory: ${BUILD_DIR}. Please remove it manually."
       fi
   else
       log_message "Build directory already removed or was not created."
   fi
}

# --- Główna Egzekucja ---
main() {
   log_message "AI Command Center - Advanced Termux Assembly Script - v${SCRIPT_VERSION}"
   log_message "Starting ADVANCED project generation..."

   check_dependencies

   BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/${BUILD_DIR_BASE}_XXXXXX")
   if [ -z "${BUILD_DIR}" ] || [ ! -d "${BUILD_DIR}" ]; then
       error_message "Failed to create temporary build directory. Check permissions and space in ${TMPDIR:-/tmp}."
       exit 1
   fi
   log_message "Using temporary build directory: ${BUILD_DIR}"

   cleanup_on_exit() {
       if [ -n "${BUILD_DIR}" ] && [ -d "${BUILD_DIR}" ]; then
           log_message "Performing cleanup due to script exit..."
           cleanup
       fi
   }
   trap cleanup_on_exit EXIT SIGINT SIGTERM
   
   create_project_structure
   create_backend_files
   create_frontend_files
   create_docs_files
   create_scripts_files
   create_vscode_config
   create_config_files
   create_root_level_files

   package_project

   log_message "AI Command Center ADVANCED project generation complete!"
   log_message "The ZIP archive '${ZIP_FILENAME}' is ready for Visual Studio Code."
   log_message "To use: Unzip '${ZIP_FILENAME}' and open the '${PROJECT_NAME}' folder in VS Code."
}

# Uruchomienie funkcji głównej
main

exit
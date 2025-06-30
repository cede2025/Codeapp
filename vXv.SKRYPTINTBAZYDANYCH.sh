#!/bin/bash

# AI Command Center - Database Persistence Integration Script
# Wersja: 1.0.0
# Autor: AI Assistant
# Opis: Ten skrypt automatyzuje refaktoryzację kluczowych serwisów
#       (MonitoringService, SystemWisdomEngine), aby zapisywały i odczytywały
#       dane z bazy danych PostgreSQL zamiast z pamięci.

# --- Konfiguracja Skryptu ---
set -euo pipefail

# --- Zmienne Projektowe ---
readonly BACKEND_PATH="backend"
readonly APP_PATH="${BACKEND_PATH}/app"
readonly DB_MODELS_FILE="${APP_PATH}/db/models.py"
readonly MONITORING_SERVICE_FILE="${APP_PATH}/monitoring/monitoring_service.py" # Nowy plik
readonly WISDOM_ENGINE_FILE="${APP_PATH}/services/wisdom_engine.py"
readonly MAIN_PY_FILE="${APP_PATH}/main.py"
readonly AI_AGENT_FILE="${APP_PATH}/core/ai_agent.py"

# --- Funkcje Pomocnicze ---
log_message() { echo -e "\033[1;34m[INFO]\033[0m $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
success_message() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warning_message() { echo -e "\033[1;33m[WARNING]\033[0m $1"; }
error_message() { echo -e "\033[1;31m[ERROR]\033[0m $1" >&2; }
confirm() { read -p "$1 (y/N): " -n 1 -r; echo; [[ $REPLY =~ ^[Yy]$ ]]; }

# --- Definicje Kodu do Wstrzyknięcia ---

read -r -d '' DB_MODELS_APPEND_CODE <<'EOF'

# -- Models for Monitoring and Wisdom Engine --

class MonitoringEvent(Base, TimestampMixin):
    __tablename__ = "monitoring_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON)

class MonitoringMetric(Base, TimestampMixin):
    __tablename__ = "monitoring_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String, index=True)
    metric_value: Mapped[float] = mapped_column(Float)

class KnowledgeBaseSolution(Base, TimestampMixin):
    __tablename__ = "knowledge_base_solutions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    problem_signature: Mapped[str] = mapped_column(String, unique=True, index=True)
    solution_details: Mapped[str] = mapped_column(Text)
    effectiveness_score: Mapped[float] = mapped_column(Float, default=1.0)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
EOF

read -r -d '' MONITORING_SERVICE_DB_CODE <<'EOF'
# backend/app/monitoring/monitoring_service.py

import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy import func

from ..db.models import MonitoringEvent, MonitoringMetric

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.get_session = session_factory

    async def log_event(self, event_type: str, details: Dict[str, Any]):
        async with self.get_session() as db:
            event = MonitoringEvent(event_type=event_type, details=details)
            db.add(event)
            await db.commit()
        logger.debug(f"[Monitoring-DB] Logged event: {event_type}")

    async def log_metric(self, metric_name: str, value: float):
        async with self.get_session() as db:
            metric = MonitoringMetric(metric_name=metric_name, metric_value=value)
            db.add(metric)
            await db.commit()
        logger.debug(f"[Monitoring-DB] Logged metric: {metric_name} = {value}")

    async def generate_report(self) -> Dict[str, Any]:
        async with self.get_session() as db:
            # Summary
            total_tasks_stmt = select(func.count(MonitoringEvent.id)).where(MonitoringEvent.event_type == 'TASK_STARTED')
            total_tasks = (await db.execute(total_tasks_stmt)).scalar_one_or_none() or 0
            
            success_tasks_stmt = select(func.count(MonitoringEvent.id)).where(MonitoringEvent.event_type.like('%COMPLETED%'))
            successful_tasks = (await db.execute(success_tasks_stmt)).scalar_one_or_none() or 0

            # Research Analytics
            research_activations_stmt = select(func.count(MonitoringEvent.id)).where(MonitoringEvent.event_type == 'RESEARCH_ACTIVATED')
            research_activations = (await db.execute(research_activations_stmt)).scalar_one_or_none() or 0
            
            avg_confidence_stmt = select(func.avg(MonitoringMetric.metric_value)).where(MonitoringMetric.metric_name == 'research_confidence')
            avg_confidence = (await db.execute(avg_confidence_stmt)).scalar_one_or_none() or 0.0

            return {
                "summary": {"total_tasks_processed": total_tasks, "success_rate": (successful_tasks / total_tasks) if total_tasks > 0 else 0},
                "research_analytics": {"activation_count": research_activations, "avg_confidence_score": avg_confidence},
            }
EOF

read -r -d '' WISDOM_ENGINE_DB_CODE <<'EOF'
# backend/app/services/wisdom_engine.py

import logging
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert

from ..db.models import KnowledgeBaseSolution
from ..models_schemas.ai_core import AITaskResult, StrategicInsight, AITaskStatus, AITaskDefinition
from ..monitoring.monitoring_service import MonitoringService

logger = logging.getLogger(__name__)

class SystemWisdomEngine:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], monitoring_service: MonitoringService):
        self.get_session = session_factory
        self.monitoring = monitoring_service
        self.strategic_insights: List[StrategicInsight] = []
        logger.info("[WisdomEngine-DB] Initialized.")

    async def learn_from_history(self, history: List[AITaskResult]):
        logger.critical("--- [WisdomEngine-DB] Starting learning cycle from history ---")
        async with self.get_session() as db:
            for result in history:
                if result.final_status == AITaskStatus.COMPLETED_AFTER_RESEARCH and result.failure_reason and result.solution_used:
                    # Użyj "upsert" (INSERT ... ON CONFLICT DO UPDATE)
                    stmt = insert(KnowledgeBaseSolution).values(
                        problem_signature=result.failure_reason,
                        solution_details=result.solution_used,
                        effectiveness_score=1.0, # Domyślnie
                        usage_count=1
                    ).on_conflict_do_update(
                        index_elements=['problem_signature'],
                        set_=dict(
                            solution_details=result.solution_used,
                            effectiveness_score=KnowledgeBaseSolution.effectiveness_score * 0.9 + 0.1, # Przykład aktualizacji
                            usage_count=KnowledgeBaseSolution.usage_count + 1
                        )
                    )
                    await db.execute(stmt)
            await db.commit()
        await self.load_insights_from_db()

    async def load_insights_from_db(self):
        logger.info("[WisdomEngine-DB] Loading strategic insights from database...")
        async with self.get_session() as db:
            stmt = select(KnowledgeBaseSolution).where(KnowledgeBaseSolution.effectiveness_score > 0.7)
            results = await db.execute(stmt)
            solutions = results.scalars().all()

            new_insights = []
            for sol in solutions:
                keyword = sol.problem_signature.split(":")[-1]
                insight = StrategicInsight(
                    problem_signature=sol.problem_signature,
                    trigger=lambda task, k=keyword: k in task.description,
                    action=lambda task, s=sol.solution_details: setattr(task, 'proactive_hint', s) or task,
                    description=f"If task mentions '{keyword}', proactively suggest solution."
                )
                new_insights.append(insight)
            
            self.strategic_insights = new_insights
            logger.info(f"[WisdomEngine-DB] Loaded {len(self.strategic_insights)} strategic insights.")

    async def predict_and_prepare(self, task: AITaskDefinition) -> AITaskDefinition:
        if not self.strategic_insights:
            await self.load_insights_from_db() # Załaduj, jeśli puste

        for insight in self.strategic_insights:
            if insight.trigger(task):
                await self.monitoring.log_event("WISDOM_APPLIED", {"task_id": task.task_id, "insight": insight.description})
                return insight.action(task)
        return task
EOF

# --- Główna Logika Skryptu ---

main() {
    log_message "AI Command Center - Database Persistence Integration"
    echo "-----------------------------------------------------------------"
    
    if ! confirm "This script will modify core application files to use the database for state. It's recommended to have a backup. Continue?"; then
        log_message "Operation cancelled by user."
        exit 0
    fi

    # Krok 1: Weryfikacja
    log_message "Verifying file structure..."
    for file in "$DB_MODELS_FILE" "$WISDOM_ENGINE_FILE" "$MAIN_PY_FILE" "$AI_AGENT_FILE"; do
        if [ ! -f "$file" ]; then
            error_message "Required file not found: $file. Please ensure the project structure is correct."
            exit 1
        fi
    done
    success_message "File structure verified."

    # Krok 2: Rozszerzenie modeli DB
    log_message "Appending new models to '${DB_MODELS_FILE}'..."
    # Sprawdź, czy kod już istnieje, aby uniknąć duplikacji
    if ! grep -q "class MonitoringEvent" "$DB_MODELS_FILE"; then
        echo -e "\n$DB_MODELS_APPEND_CODE" >> "$DB_MODELS_FILE"
        success_message "Database models for Monitoring and Wisdom appended."
    else
        warning_message "Monitoring and Wisdom models already exist in '${DB_MODELS_FILE}'. Skipping."
    fi

    # Krok 3: Przepisanie serwisów
    log_message "Replacing in-memory services with DB-aware versions..."
    mkdir -p "$(dirname "$MONITORING_SERVICE_FILE")"
    # Zastąpienie starych plików
    mv "$WISDOM_ENGINE_FILE" "$WISDOM_ENGINE_FILE.bak"
    echo "$WISDOM_ENGINE_DB_CODE" > "$WISDOM_ENGINE_FILE"
    # Utworzenie nowego pliku monitoringu
    echo "$MONITORING_SERVICE_DB_CODE" > "$MONITORING_SERVICE_FILE"
    success_message "Replaced WisdomEngine and created DB-aware MonitoringService."

    # Krok 4: Modyfikacja main.py
    log_message "Updating '${MAIN_PY_FILE}' to manage new service lifecycles..."
    cp "$MAIN_PY_FILE" "$MAIN_PY_FILE.bak"
    # Użyj awk do bardziej złożonej edycji - zastąp cały blok lifespan
    awk '
        BEGIN { p = 1 }
        /@asynccontextmanager/ { print; print "async def lifespan(app: FastAPI):"; p = 0; next }
        /yield/ {
            print "    # -- Initialize DB-aware services --"
            print "    app_lifespan_globals[\"monitoring_service\"] = MonitoringService(AsyncSessionLocal)"
            print "    app_lifespan_globals[\"wisdom_engine\"] = SystemWisdomEngine(AsyncSessionLocal, app_lifespan_globals[\"monitoring_service\"])"
            print "    await app_lifespan_globals[\"wisdom_engine\"].load_insights_from_db() # Initial load"
            print "    logger.info(\"DB-aware services (Monitoring, Wisdom) initialized.\")"
            print ""
            print "    # Initialize Main AI Agent with DB-aware services"
            print "    app_lifespan_globals[\"main_ai_agent\"] = MainAIAgent("
            print "        openrouter_client=app_lifespan_globals[\"openrouter_client\"],"
            print "        wisdom_engine=app_lifespan_globals[\"wisdom_engine\"],"
            print "        monitoring_service=app_lifespan_globals[\"monitoring_service\"]"
            print "    )"
            print "    logger.info(\"Main AI Agent initialized with new dependencies.\")"
            print ""
            print "    yield # Application runs here"
            print ""
            print "    # Shutdown"
            print "    logger.info(f\"Shutting down {settings.APP_NAME}...\")"
            print "    await app_lifespan_globals[\"openrouter_client\"].close()"
            print "    logger.info(\"Shutdown complete.\")"
            p = 1; next
        }
        p { print }
    ' "$MAIN_PY_FILE.bak" > "$MAIN_PY_FILE"
    # Dodaj nowe zależności do DI
    {
        echo "app.dependency_overrides[MonitoringService] = lambda: app_lifespan_globals[\"monitoring_service\"]"
        echo "app.dependency_overrides[SystemWisdomEngine] = lambda: app_lifespan_globals[\"wisdom_engine\"]"
    } >> "$MAIN_PY_FILE"
    success_message "Updated '${MAIN_PY_FILE}' with new service initializations."

    # Krok 5: Modyfikacja ai_agent.py
    log_message "Updating '${AI_AGENT_FILE}' to use dependency injection..."
    cp "$AI_AGENT_FILE" "$AI_AGENT_FILE.bak"
    # Zmiana konstruktora agenta
    sed -i.tmp 's/def __init__(self, openrouter_client: OpenRouterClient):/def __init__(self, openrouter_client: OpenRouterClient, wisdom_engine: "SystemWisdomEngine", monitoring_service: "MonitoringService"):/' "$AI_AGENT_FILE"
    sed -i.tmp 's/self\.wisdom_engine = wisdom_engine/self.wisdom_engine = wisdom_engine\n        self.monitoring = monitoring_service/' "$AI_AGENT_FILE"
    rm "$AI_AGENT_FILE.tmp"
    success_message "Updated '${AI_AGENT_FILE}' constructor."

    # Krok 6: Instrukcje końcowe
    echo
    warning_message "------------------------- IMPORTANT NEXT STEPS -------------------------"
    warning_message "The code has been refactored, but the database schema is now out of date."
    warning_message "You MUST generate and apply a new database migration."
    warning_message "Run the following commands from the 'backend/' directory:"
    echo
    echo -e "  \033[1;33m1. alembic revision --autogenerate -m \"Add persistence for monitoring and wisdom\"\033[0m"
    echo -e "  \033[1;33m2. alembic upgrade head\033[0m"
    echo
    warning_message "After running the migration, your application will be ready."
    warning_message "----------------------------------------------------------------------"
    echo
    success_message "Refactoring for database persistence complete!"
}

# Uruchomienie funkcji głównej
main
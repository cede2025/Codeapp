# 🧠 AI Command Center

W pełni skonteneryzowana platforma do orkiestracji agentów AI, gotowa do wdrożenia w dowolnym środowisku wspierającym Dockera.

## 🚀 Pierwsze Uruchomienie

Ten projekt jest przeznaczony do uruchomienia za pomocą Docker Compose.

1.  **Skopiuj plik środowiskowy:**
    ```bash
    cp .env.example .env
    ```

2.  **Zbuduj i uruchom kontenery:**
    ```bash
    docker-compose up --build -d
    ```
    Opcja `-d` uruchamia kontenery w tle.

3.  **Sprawdź dostępne usługi:**
    -   **Frontend (UI):** `http://localhost:3000`
    -   **Backend (API Docs):** `http://localhost:8000/docs`
    -   **Grafana (Monitoring):** `http://localhost:3001` (login: `admin`/`admin`)
    -   **Prometheus:** `http://localhost:9090`

4.  **Aby zatrzymać aplikację:**
    ```bash
    docker-compose down
    ```

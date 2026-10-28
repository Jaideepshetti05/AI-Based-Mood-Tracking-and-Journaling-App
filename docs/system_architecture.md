# System architecture

This diagram describes the implemented local application. The browser UI is
served by FastAPI from `backend/app/static/index.html`; the Flutter client is
an independent API consumer. The optional transformer and fallback are invoked
by `AnalysisService`, while persistence and aggregation remain in the backend.

```mermaid
flowchart LR
    U[User]
    subgraph Clients
        B[Browser UI<br/>FastAPI static HTML/JS]
        F[Flutter client<br/>mobile/lib]
    end
    subgraph API[FastAPI backend]
        R[API routes]
        V[Pydantic validation<br/>and X-User-ID loopback guard]
        S[Journal and mood services]
        A[AnalysisService]
        P[PersonalizationEngine]
        I[Weekly/monthly insights]
        E[CSV export and deletion]
    end
    subgraph NLP[NLP analysis]
        D[VADER sentiment<br/>compound score]
        T[Optional transformer<br/>emotion classification]
        H[Heuristic emotion fallback]
    end
    DB[(SQLite via SQLAlchemy)]
    C[Selected mood vs predicted<br/>emotion and feedback]
    CH[Mood, sentiment, and<br/>emotion visualizations]

    U --> B
    U --> F
    B -->|HTTP JSON| R
    F -->|HTTP JSON| R
    R --> V
    V --> S
    S --> A
    A --> D
    A --> T
    A --> H
    D --> A
    T --> A
    H --> A
    S --> P
    P --> S
    S --> DB
    R --> I
    I --> DB
    R --> E
    E --> DB
    S --> C
    C --> B
    C --> F
    I --> CH
    CH --> B
    CH --> F
    R -->|API responses| B
    R -->|API responses| F
```

## Main data flow

1. A client submits an entry or analysis request.
2. FastAPI validates the payload and resolves the local user namespace.
3. `AnalysisService` computes VADER sentiment and either transformer or
   heuristic emotion output.
4. `PersonalizationEngine` optionally adjusts the emotion using prior feedback.
5. SQLAlchemy persists journal, analysis, correction, and preference records in
   SQLite.
6. The API returns the saved result, or aggregates persisted records for
   insights and export.

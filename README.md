# SME-to-Session Scheduling Agent

A deployable AI scheduling copilot for Ops/Curriculum teams managing high-volume live learning sessions.

## Product thesis

The system does not simply "pick an SME." It converts a messy weekly scheduling problem into a constrained decision workflow:

**Ingest → Validate → Candidate generation → Hard constraints → Semantic expertise reasoning → Fairness optimization → Conflict ranking → Explain → Human approval**

The LLM is deliberately bounded to the part where language models are useful: interpreting adjacent expertise and generating concise reasoning. Safety-critical scheduling rules remain deterministic.

## What the prototype demonstrates

- Weekly session + SME ingestion using synthetic data.
- Timezone-aware availability validation.
- Hard constraints: availability, training level, dropout status and double-booking.
- Topic expertise matching with deterministic fallback and optional LLM semantic reasoning.
- Score breakdown across expertise, performance, fairness, preference, availability and rotation.
- Rolling 4-week workload fairness.
- Near-tie detection and alternative candidates.
- Unfilled-session detection and high-severity escalation.
- Scenario simulator:
  - Normal week
  - SME dropout
  - No qualified SME
  - Double booking
  - Workload imbalance
- Assignment inspector with "why this SME?" and alternatives.
- Human override with mandatory reason.
- Approval gate that blocks unresolved high-severity conflicts.
- Agent execution trace.
- Audit trail.
- Google Sheets / Google Calendar adapter-ready architecture.
- Vercel-ready FastAPI deployment.

## LLM architecture

If `OPENAI_API_KEY` is present, the backend uses the OpenAI Responses API for semantic expertise evaluation. The default model is configurable through `OPENAI_MODEL`.

If the key is absent or the model call fails, the system automatically falls back to the deterministic topic ontology. This keeps the demo reproducible and prevents the scheduling engine from becoming dependent on an external model.

The LLM **cannot**:
- override availability;
- override training requirements;
- assign a dropped-out SME;
- create a double booking;
- approve a schedule;
- silently bypass a hard constraint.

## Environment variables

```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-5.6-luna
```


## Local run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn api.index:app --reload
```

Open `http://localhost:8000`.

## Vercel

```bash
npm i -g vercel
vercel login
vercel
vercel --prod
```


## API surface

- `GET /api/health`
- `GET /api/data`
- `POST /api/run`
- `GET /api/draft`
- `GET /api/sessions/{session_id}`
- `GET /api/conflicts`
- `GET /api/smes/workload`
- `POST /api/override`
- `POST /api/approve`
- `GET /api/audit`

## Production evolution

1. Replace synthetic adapters with authenticated Google Sheets / Calendar connectors.
2. Persist runs, approvals and overrides in Postgres.
3. Add scheduled weekly triggers and dropout webhooks.
4. Add a learned performance model from session quality and learner feedback.
5. Move from weighted ranking to constrained optimization when volume justifies it.
6. Add RBAC and immutable audit logging.

## Success metrics

- ≥85% sessions auto-assigned.
- 0% hard-constraint violations.
- ≥60% reduction in scheduling effort.
- <20% human override rate after tuning.
- ≤2 assignment spread across comparable SMEs where feasible.
- ≥95% recall for high-risk conflicts.
- No regression in learner session quality.


## Resolution and approval workflow

The prototype intentionally blocks approval when a high-severity gap remains. Ops can now resolve a blocking session directly from the exception queue using one of two explicit actions:

- **Manual assign with expertise exception**: preserves availability, training and double-booking guardrails while recording the human exception as a medium-risk decision.
- **Defer session**: removes the session from the publishable schedule and records the operational rationale.

Once all high-severity issues are resolved, the approval CTA becomes active and records the human approval in the audit trail.

The UI also normalizes SME skills stored as either arrays or comma-separated strings, so roster cards and SME profiles render reliably against the synthetic data.

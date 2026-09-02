from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy
import json
import os
import re

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI(
    title="SME-to-Session Scheduling Agent",
    version="1.0.0",
    description="AI scheduling copilot with deterministic hard constraints, semantic expertise reasoning, fairness optimization and human approval."
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PUBLIC = ROOT / "public"

with open(DATA / "synthetic_week.json", "r", encoding="utf-8") as f:
    BASE = json.load(f)

state = {
    "draft": None,
    "approved": False,
    "audit": [],
    "run_id": None,
    "scenario": "Normal week",
}

TOPIC_ALIASES = {
    "product strategy": {"product management", "product analytics", "business strategy"},
    "product analytics": {"product management", "data analytics", "sql", "analytics"},
    "sql": {"data analytics", "analytics", "python"},
    "system design": {"backend", "software architecture", "distributed systems"},
    "python": {"backend", "data analytics", "machine learning"},
    "behavioral interview": {"interview prep", "career coaching", "mock interview"},
    "mock interview": {"interview prep", "career coaching", "behavioral interview"},
    "data storytelling": {"data analytics", "product analytics", "business intelligence"},
}

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def dt(value):
    x = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return x.replace(tzinfo=None) if x.tzinfo else x

def overlap(a_start, a_end, b_start, b_end):
    return max(dt(a_start), dt(b_start)) < min(dt(a_end), dt(b_end))

def duration_hours(item):
    return max(0.5, (dt(item["end"]) - dt(item["start"])).total_seconds() / 3600)

def normalise(value):
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;/|]", str(value))
    return {x.strip().lower() for x in raw if x and x.strip()}

def semantic_fallback(required, skills):
    req = normalise(required)
    skill_set = normalise(skills)
    exact = sorted(req & skill_set)
    if exact:
        return {
            "fit": "direct",
            "confidence": 0.98,
            "fit_score": 1.0,
            "matched_skills": exact,
            "missing_skills": [],
            "reason": "Direct topic coverage."
        }
    for r in req:
        adjacent = TOPIC_ALIASES.get(r, set()) & skill_set
        if adjacent:
            return {
                "fit": "adjacent",
                "confidence": 0.82,
                "fit_score": 0.78,
                "matched_skills": sorted(adjacent),
                "missing_skills": [],
                "reason": f"Adjacent expertise through {', '.join(sorted(adjacent))}."
            }
    return {
        "fit": "none",
        "confidence": 0.96,
        "fit_score": 0.0,
        "matched_skills": [],
        "missing_skills": sorted(req),
        "reason": "No credible topic relationship found."
    }

def llm_enabled():
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None

def llm_semantic_batch(pairs):
    """
    LLM is deliberately restricted to semantic expertise interpretation.
    It cannot approve availability, training, dropout, double-booking or fairness.
    """
    if not pairs or not llm_enabled():
        return {}

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = """You are the semantic expertise layer of an SME scheduling system.
Evaluate ONLY whether each SME has direct, adjacent, or no expertise for the session topic.
Do not reason about availability, workload, training, fairness, or scheduling.
Return ONLY valid JSON: {"results":[{"pair_id":"...","fit":"direct|adjacent|none","confidence":0.0,"fit_score":0.0,"matched_skills":[],"missing_skills":[],"reason":"short"}]}
Use fit_score 1.0 for direct, 0.70-0.90 for credible adjacent expertise, and 0.0 for no credible fit.
"""

    payload = []
    for p in pairs:
        payload.append({
            "pair_id": p["pair_id"],
            "session_topic": p["topic"],
            "sme_name": p["sme_name"],
            "sme_skills": p["skills"],
        })

    try:
        response = client.responses.create(
            model=model,
            input=prompt + "\n\nPAIR DATA:\n" + json.dumps(payload),
        )
        text = getattr(response, "output_text", "") or ""
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        parsed = json.loads(match.group(0))
        return {x["pair_id"]: x for x in parsed.get("results", [])}
    except Exception as exc:
        state["audit"].append({
            "time": now_iso(),
            "event": "llm_fallback",
            "detail": str(exc)[:180],
        })
        return {}

def workload_counts(history, week_start):
    cutoff = dt(week_start) - timedelta(days=28)
    counts = {}
    for row in history:
        if dt(row["date"]) >= cutoff:
            counts[row["sme_id"]] = counts.get(row["sme_id"], 0) + 1
    return counts

def validate_assignment(session, sme, used):
    if not sme:
        return False, "No SME selected."
    if sme.get("status") == "dropped_out":
        return False, "SME is marked as dropped out."
    if sme["training_level"] < session["required_training_level"]:
        return False, "Training level below session requirement."
    available = any(
        slot["date"] == session["date"]
        and dt(slot["start"]) <= dt(session["start"])
        and dt(slot["end"]) >= dt(session["end"])
        for slot in sme.get("availability", [])
    )
    if not available:
        return False, "SME is not available for the full session window."
    if any(overlap(a["start"], a["end"], session["start"], session["end"]) for a in used.get(sme["id"], [])):
        return False, "SME would be double-booked."
    return True, ""

def build_candidates(sessions, smes, history, week_start):
    counts = workload_counts(history, week_start)
    max_load = max(counts.values() or [0])
    candidates_by_session = {}
    eligible_pairs = []

    for session in sessions:
        candidates = []
        for sme in smes:
            ok, hard_reason = validate_assignment(session, sme, {})
            if not ok:
                continue

            fallback = semantic_fallback(session["topic"], sme["skills"])
            # Direct expertise is a hard requirement for auto-assignment.
            # Adjacent expertise is allowed into review, never silently auto-approved.
            if fallback["fit_score"] <= 0:
                continue

            perf = sme.get("topic_performance", {}).get(
                session["topic"], sme.get("overall_performance", 0.75)
            )
            pref = 1.0 if session["mode"] in sme.get("preferred_modes", []) else 0.0
            fair = 1 - (
                counts.get(sme["id"], 0) /
                max(1, max_load + 2)
            )
            rotation = 1.0 if counts.get(sme["id"], 0) < max_load else 0.0

            pair_id = f'{session["id"]}::{sme["id"]}'
            candidates.append({
                "sme": sme,
                "pair_id": pair_id,
                "fallback": fallback,
                "performance": perf,
                "preference": pref,
                "fairness": fair,
                "rotation": rotation,
            })
            if fallback["fit"] != "direct":
                eligible_pairs.append({
                    "pair_id": pair_id,
                    "topic": session["topic"],
                    "sme_name": sme["name"],
                    "skills": sme["skills"],
                })

        candidates_by_session[session["id"]] = candidates

    llm_results = llm_semantic_batch(eligible_pairs)
    return candidates_by_session, counts, llm_results

def run_matching(scenario="Normal week"):
    sessions = deepcopy(BASE["sessions"])
    smes = deepcopy(BASE["smes"])
    history = deepcopy(BASE["assignment_history"])

    if scenario == "SME dropout":
        target = next((s for s in smes if s["id"] == "E-04"), None)
        if target:
            target["status"] = "dropped_out"
            target["availability"] = []
    elif scenario == "No qualified SME":
        target = next((s for s in sessions if s["id"] == "S-110"), None)
        if target:
            target["topic"] = "Quantum Computing"
    elif scenario == "Double booking":
        target = next((s for s in smes if s["id"] == "E-01"), None)
        if target:
            target["availability"].append({
                "date": "2026-09-08",
                "start": "2026-09-08T10:30:00+05:30",
                "end": "2026-09-08T13:30:00+05:30"
            })
        # Adds an intentionally overlapping session.
        sessions.append({
            "id": "S-111",
            "date": "2026-09-07",
            "start": "2026-09-07T10:30:00+05:30",
            "end": "2026-09-07T11:30:00+05:30",
            "topic": "Product Strategy",
            "mode": "live",
            "required_training_level": 2
        })
    elif scenario == "Workload imbalance":
        history += [{"date": "2026-09-01", "sme_id": "E-01"}] * 5

    week_start = BASE["week_start"]
    candidates_by_session, counts, llm_results = build_candidates(
        sessions, smes, history, week_start
    )

    draft = []
    conflicts = []
    used = {}
    trace = [
        {"step": "INGEST", "status": "complete", "detail": f"Loaded {len(sessions)} sessions and {len(smes)} SMEs"},
        {"step": "VALIDATE", "status": "complete", "detail": "Validated time windows, training levels and source fields"},
        {"step": "CANDIDATES", "status": "complete", "detail": "Generated eligible candidate sets"},
        {"step": "SEMANTIC FIT", "status": "complete", "detail": "Evaluated adjacent expertise with LLM when configured"},
        {"step": "OPTIMIZE", "status": "complete", "detail": "Scored expertise, performance, fairness, preference and rotation"},
        {"step": "EXCEPTIONS", "status": "complete", "detail": "Ranked gaps, mismatches, ties and fairness risks"},
        {"step": "HUMAN REVIEW", "status": "ready", "detail": "Draft is ready for approval or override"},
    ]

    # Score weights are intentionally visible and stable.
    for session in sorted(sessions, key=lambda x: (x["date"], x["start"])):
        candidates = []
        for c in candidates_by_session.get(session["id"], []):
            sme = c["sme"]
            fit = c["fallback"]["fit_score"]
            semantic = llm_results.get(c["pair_id"])
            if semantic:
                fit = float(semantic.get("fit_score", fit))
                fit_reason = semantic.get("reason", c["fallback"]["reason"])
                fit_type = semantic.get("fit", c["fallback"]["fit"])
            else:
                fit_reason = c["fallback"]["reason"]
                fit_type = c["fallback"]["fit"]

            # A candidate that is adjacent expertise remains review-only.
            projected_load = counts.get(sme["id"], 0) + len(used.get(sme["id"], []))
            fair = 1 - (projected_load / max(1, max(counts.values() or [0]) + 2))
            score = (
                0.35 * fit +
                0.20 * c["performance"] +
                0.20 * fair +
                0.10 * c["preference"] +
                0.10 * (1 if validate_assignment(session, sme, used)[0] else 0) +
                0.05 * c["rotation"]
            )
            candidates.append({
                **c,
                "fit": fit,
                "fit_type": fit_type,
                "fit_reason": fit_reason,
                "fairness": fair,
                "score": score,
            })

        # Do not allow an SME already used in an overlapping session.
        candidates = [
            c for c in candidates
            if validate_assignment(session, c["sme"], used)[0]
        ]
        candidates.sort(key=lambda x: (-x["score"], x["sme"]["id"]))

        if not candidates:
            draft.append({
                "session_id": session["id"],
                "sme_id": None,
                "sme_name": None,
                "status": "unfilled",
                "score": 0,
                "topic": session["topic"],
                "date": session["date"],
                "start": session["start"],
                "end": session["end"],
                "mode": session["mode"],
                "duration_hours": duration_hours(session),
                "reason": "No qualified and available SME survives the hard constraints.",
                "alternatives": [],
                "hard_constraints": ["availability", "training level", "expertise", "no double-booking"],
            })
            conflicts.append({
                "session_id": session["id"],
                "type": "unfilled",
                "severity": "high",
                "title": "No feasible assignment",
                "reason": "No SME satisfies the hard constraints. Move the session, widen the qualified pool, or manually intervene.",
                "action": "Escalate to Ops/Curriculum",
            })
            continue

        top = candidates[0]
        tie = len(candidates) > 1 and abs(candidates[0]["score"] - candidates[1]["score"]) < 0.025
        review_only = top["fit_type"] != "direct"
        status = "needs_review" if tie or review_only else "matched"

        alternatives = []
        for alt in candidates[1:4]:
            alternatives.append({
                "sme_id": alt["sme"]["id"],
                "sme_name": alt["sme"]["name"],
                "score": round(alt["score"], 3),
                "fit": alt["fit_type"],
                "performance": round(alt["performance"], 2),
                "fairness": round(alt["fairness"], 2),
            })

        reason_parts = [
            f'{int(top["fit"] * 100)}% topic-fit',
            f'{top["performance"]:.2f} historical performance',
            f'{counts.get(top["sme"]["id"], 0)} assignments in rolling window',
        ]
        if top["preference"]:
            reason_parts.append("mode preference aligned")
        if review_only:
            reason_parts.append("adjacent expertise requires review")
        if tie:
            reason_parts.append("near-tie with next-best candidate")

        item = {
            "session_id": session["id"],
            "sme_id": top["sme"]["id"],
            "sme_name": top["sme"]["name"],
            "status": status,
            "score": round(top["score"], 3),
            "topic": session["topic"],
            "date": session["date"],
            "start": session["start"],
            "end": session["end"],
            "mode": session["mode"],
            "duration_hours": round(duration_hours(session), 2),
            "reason": "; ".join(reason_parts),
            "fit_type": top["fit_type"],
            "fit_confidence": round(
                float((llm_results.get(top["pair_id"]) or top["fallback"]).get("confidence", 0.9)), 2
            ),
            "score_breakdown": {
                "topic_expertise": round(0.35 * top["fit"], 3),
                "historical_performance": round(0.20 * top["performance"], 3),
                "fairness": round(0.20 * top["fairness"], 3),
                "preference": round(0.10 * top["preference"], 3),
                "availability": 0.10,
                "rotation": round(0.05 * top["rotation"], 3),
            },
            "alternatives": alternatives,
            "hard_constraints": ["availability passed", "training level passed", "no double-booking"],
        }
        draft.append(item)
        used.setdefault(top["sme"]["id"], []).append(session)
        counts[top["sme"]["id"]] = counts.get(top["sme"]["id"], 0) + 1

        if review_only:
            conflicts.append({
                "session_id": session["id"],
                "type": "expertise_mismatch",
                "severity": "medium",
                "title": "Adjacent expertise",
                "reason": top["fit_reason"],
                "action": "Curriculum review recommended before approval",
            })
        if tie:
            conflicts.append({
                "session_id": session["id"],
                "type": "tie",
                "severity": "medium",
                "title": "Near-tie decision",
                "reason": "Top candidates are within 2.5 percentage points. Human context can break the tie.",
                "action": "Review the alternatives in the assignment inspector",
            })

    final_counts = {s["id"]: counts.get(s["id"], 0) for s in smes}
    active_counts = [v for k, v in final_counts.items() if any(s["id"] == k and s.get("status") != "dropped_out" for s in smes)]
    spread = max(active_counts) - min(active_counts) if active_counts else 0
    if spread >= 3:
        conflicts.append({
            "session_id": None,
            "type": "fairness",
            "severity": "medium",
            "title": "Workload concentration",
            "reason": f"Rolling workload spread is {spread} assignments. The draft favors expertise, but the imbalance should be reviewed.",
            "action": "Use the workload view before approval",
        })

    auto_assigned = sum(1 for x in draft if x["status"] == "matched")
    needs_review = sum(1 for x in draft if x["status"] == "needs_review")
    unfilled = sum(1 for x in draft if x["status"] == "unfilled")

    state["scenario"] = scenario
    state["run_id"] = f"run-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    state["draft"] = {
        "run_id": state["run_id"],
        "scenario": scenario,
        "week_start": week_start,
        "generated_at": now_iso(),
        "assignments": draft,
        "conflicts": conflicts,
        "workload": final_counts,
        "workload_spread": spread,
        "summary": {
            "sessions": len(draft),
            "auto_assigned": auto_assigned,
            "needs_review": needs_review,
            "unfilled": unfilled,
            "coverage": round((auto_assigned + needs_review) / max(1, len(draft)), 2),
            "review_rate": round(needs_review / max(1, len(draft)), 2),
        },
        "agent_trace": trace,
        "llm": {
            "enabled": llm_enabled(),
            "provider": "OpenAI" if llm_enabled() else "Deterministic fallback",
            "model": os.getenv("OPENAI_MODEL", "gpt-5.6-luna") if llm_enabled() else None,
        },
        "approved": False,
    }
    state["approved"] = False
    state["audit"].append({
        "time": now_iso(),
        "event": "draft_generated",
        "run_id": state["run_id"],
        "scenario": scenario,
        "summary": state["draft"]["summary"],
    })
    return state["draft"]

@app.get("/")
def root():
    return FileResponse(PUBLIC / "index.html")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "agent": "sme-scheduler",
        "version": "1.0.0",
        "llm": {
            "enabled": llm_enabled(),
            "provider": "OpenAI" if llm_enabled() else "Deterministic fallback",
            "model": os.getenv("OPENAI_MODEL", "gpt-5.6-luna") if llm_enabled() else None,
        },
        "integrations": {
            "google_sheets": "adapter-ready",
            "google_calendar": "adapter-ready",
        },
    }

@app.get("/api/data")
def get_data():
    return {
        "week_start": BASE["week_start"],
        "sessions": BASE["sessions"],
        "smes": BASE["smes"],
        "assignment_history": BASE["assignment_history"],
        "data_sources": [
            {"name": "Synthetic weekly schedule", "status": "connected", "type": "demo"},
            {"name": "Google Sheets", "status": "adapter-ready", "type": "production"},
            {"name": "Google Calendar", "status": "adapter-ready", "type": "production"},
        ],
    }

@app.post("/api/run")
def run(payload: dict = None):
    payload = payload or {}
    scenario = payload.get("scenario", "Normal week")
    allowed = {"Normal week", "SME dropout", "No qualified SME", "Double booking", "Workload imbalance"}
    if scenario not in allowed:
        raise HTTPException(400, "Unknown scenario")
    return run_matching(scenario)

@app.get("/api/draft")
def draft():
    if not state["draft"]:
        return run_matching()
    return state["draft"]

@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    if not state["draft"]:
        run_matching()
    item = next((x for x in state["draft"]["assignments"] if x["session_id"] == session_id), None)
    if not item:
        raise HTTPException(404, "Session not found")
    return item

@app.get("/api/conflicts")
def conflicts():
    if not state["draft"]:
        run_matching()
    return state["draft"]["conflicts"]

@app.get("/api/smes/workload")
def workloads():
    if not state["draft"]:
        run_matching()
    return state["draft"]["workload"]

@app.get("/api/audit")
def audit():
    return state["audit"]

def draft_session(session_id):
    """Return the live session definition for the current scenario."""
    if not state["draft"]:
        run_matching()
    item = next((x for x in state["draft"]["assignments"] if x["session_id"] == session_id), None)
    if not item:
        return None, None
    session = next((x for x in BASE["sessions"] if x["id"] == session_id), None)
    if session is None:
        # Scenario-generated sessions are represented by the assignment row.
        session = {k: item[k] for k in ("session_id", "topic", "date", "start", "end", "mode") if k in item}
        session["id"] = session_id
        session["required_training_level"] = 2
    return item, session


def current_used(exclude_session_id=None):
    used = {}
    for item in (state["draft"] or {}).get("assignments", []):
        if item.get("session_id") == exclude_session_id or not item.get("sme_id") or item.get("status") == "deferred":
            continue
        used.setdefault(item["sme_id"], []).append(item)
    return used


def refresh_summary():
    draft = state["draft"]
    assignments = draft["assignments"]
    counts = {sid: 0 for sid in draft.get("workload", {})}
    for item in assignments:
        if item.get("sme_id") and item.get("status") != "deferred":
            counts[item["sme_id"]] = counts.get(item["sme_id"], 0) + 1
    draft["workload"] = counts
    active = [v for sid, v in counts.items() if not any(s["id"] == sid and s.get("status") == "dropped_out" for s in BASE["smes"])]
    draft["workload_spread"] = max(active) - min(active) if active else 0
    matched = sum(1 for x in assignments if x.get("status") == "matched")
    review = sum(1 for x in assignments if x.get("status") == "needs_review")
    unfilled = sum(1 for x in assignments if x.get("status") == "unfilled")
    overridden = sum(1 for x in assignments if x.get("status") == "overridden")
    deferred = sum(1 for x in assignments if x.get("status") == "deferred")
    scheduled = matched + review + overridden
    draft["summary"] = {
        "sessions": len(assignments),
        "auto_assigned": matched,
        "needs_review": review,
        "unfilled": unfilled,
        "overridden": overridden,
        "deferred": deferred,
        "coverage": round(scheduled / max(1, len(assignments)), 2),
        "review_rate": round(review / max(1, len(assignments)), 2),
    }
    return draft


def remove_conflicts_for_session(session_id):
    draft = state["draft"]
    draft["conflicts"] = [c for c in draft.get("conflicts", []) if c.get("session_id") != session_id]


def add_medium_conflict(session_id, title, reason, action):
    state["draft"]["conflicts"].append({
        "session_id": session_id,
        "type": "manual_exception",
        "severity": "medium",
        "title": title,
        "reason": reason,
        "action": action,
    })

@app.post("/api/override")
def override(payload: dict):
    if not state["draft"]:
        run_matching()
    session_id = payload.get("session_id")
    new_id = payload.get("sme_id")
    reason = (payload.get("reason") or "").strip()
    if not session_id or not new_id or not reason:
        raise HTTPException(400, "session_id, sme_id and override reason are required")

    item, session = draft_session(session_id)
    sme = next((x for x in BASE["smes"] if x["id"] == new_id), None)
    if not item or not session or not sme:
        raise HTTPException(404, "Session or SME not found")

    ok, hard_reason = validate_assignment(session, sme, current_used(session_id))
    if not ok:
        raise HTTPException(400, f"Override violates hard constraint: {hard_reason}")

    old_id = item.get("sme_id")
    if old_id and old_id != new_id:
        state["draft"]["workload"][old_id] = max(0, state["draft"]["workload"].get(old_id, 0) - 1)
    state["draft"]["workload"][new_id] = state["draft"]["workload"].get(new_id, 0) + (0 if old_id == new_id else 1)

    item["sme_id"] = new_id
    item["sme_name"] = sme["name"]
    item["status"] = "overridden"
    item["reason"] = f"Human override: {reason}"
    item["override_reason"] = reason
    item["fit_type"] = item.get("fit_type") or "manual"
    remove_conflicts_for_session(session_id)
    add_medium_conflict(session_id, "Human override", f"Ops manually selected {sme['name']}: {reason}", "Review the override in the audit trail before approval")
    refresh_summary()
    state["audit"].append({
        "time": now_iso(),
        "event": "override_staged",
        "session_id": session_id,
        "sme_id": new_id,
        "reason": reason,
    })
    return state["draft"]


@app.post("/api/resolve")
def resolve_exception(payload: dict):
    """Resolve a blocking gap through an explicit Ops decision."""
    if not state["draft"]:
        run_matching()
    session_id = payload.get("session_id")
    action = payload.get("action")
    new_id = payload.get("sme_id")
    reason = (payload.get("reason") or "").strip()
    if not session_id or action not in {"manual_assign", "defer"} or not reason:
        raise HTTPException(400, "session_id, action and resolution reason are required")

    item, session = draft_session(session_id)
    if not item or not session:
        raise HTTPException(404, "Session not found")
    if item.get("status") != "unfilled":
        raise HTTPException(400, "This session no longer has a blocking gap.")

    if action == "defer":
        item["status"] = "deferred"
        item["reason"] = f"Deferred by Ops: {reason}"
        item["resolution_reason"] = reason
        item["sme_id"] = None
        item["sme_name"] = None
        remove_conflicts_for_session(session_id)
        add_medium_conflict(session_id, "Session deferred", f"Ops removed this session from the publishable schedule: {reason}", "Confirm the deferred session is communicated to the program team")
        event = "session_deferred"
        audit_sme = None
    else:
        if not new_id:
            raise HTTPException(400, "Select an SME for manual assignment.")
        sme = next((x for x in BASE["smes"] if x["id"] == new_id), None)
        if not sme:
            raise HTTPException(404, "SME not found")
        ok, hard_reason = validate_assignment(session, sme, current_used(session_id))
        if not ok:
            raise HTTPException(400, f"Manual assignment violates hard constraint: {hard_reason}")
        item["sme_id"] = new_id
        item["sme_name"] = sme["name"]
        item["status"] = "overridden"
        item["score"] = 0.5
        item["fit_type"] = "manual_exception"
        item["fit_confidence"] = 0.5
        item["reason"] = f"Manual exception: {reason}"
        item["override_reason"] = reason
        item["hard_constraints"] = ["availability passed", "training level passed", "no double-booking", "expertise exception accepted by Ops"]
        remove_conflicts_for_session(session_id)
        add_medium_conflict(session_id, "Expertise exception", f"{sme['name']} was manually assigned despite the original expertise gap: {reason}", "Review the exception rationale before approval")
        event = "manual_exception_assignment"
        audit_sme = new_id

    refresh_summary()
    state["audit"].append({
        "time": now_iso(),
        "event": event,
        "session_id": session_id,
        "sme_id": audit_sme,
        "reason": reason,
    })
    return state["draft"]


@app.post("/api/approve")
def approve(payload: dict = None):
    if not state["draft"]:
        run_matching()
    draft = state["draft"]
    unresolved_high = [c for c in draft["conflicts"] if c["severity"] == "high"]
    if unresolved_high:
        raise HTTPException(409, "Resolve high-severity conflicts before approval.")

    draft["approved"] = True
    draft["approved_at"] = now_iso()
    draft["agent_trace"][-1] = {
        "step": "HUMAN REVIEW",
        "status": "approved",
        "detail": "Human approval recorded. Final write-back would happen through the Sheets/Calendar adapters."
    }
    state["approved"] = True
    state["audit"].append({
        "time": now_iso(),
        "event": "schedule_approved",
        "run_id": state["run_id"],
    })
    return draft


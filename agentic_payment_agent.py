from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agentic_framework import (
    Agent,
    CompositeAuditSink,
    Context,
    Flow,
    JsonlAuditSink,
    Step,
    StepResult,
)

import screening_core as core


class DictListAuditSink:
    """Audit sink that appends events as dicts into a caller-owned list."""

    def __init__(self, target: List[Dict[str, Any]]):
        self._target = target

    def emit(self, event) -> None:  # conforms to AuditSink protocol at runtime
        self._target.append(
            {
                "ts": event.ts,
                "job_id": event.job_id,
                "item_index": event.item_index,
                "step": event.step,
                "message": event.message,
                "data": event.data,
            }
        )


def _default_audit_path() -> str:
    return os.path.join(os.path.dirname(__file__), "audit_trail.jsonl")


def generate_explanation(payload: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    decision = result.get("decision")
    reason = result.get("reason")
    best_score = result.get("best_score")
    best_role = result.get("best_role")
    best_wl = result.get("best_wl") or {}
    breakdown = result.get("breakdown") or {}
    sanction_flag = bool(result.get("sanction_flag") or False)
    sanction_reasons = result.get("sanction_reasons") or []

    drivers: List[Tuple[str, float]] = []
    for k in ("name", "address", "dob", "country"):
        try:
            drivers.append((k, float(breakdown.get(k, 0.0))))
        except Exception:
            pass
    drivers.sort(key=lambda x: x[1], reverse=True)

    title = "Payment Screening Explanation (Agentic)"
    out = [title, "=" * len(title)]
    out.append(
        " • "
        + " | ".join(
            [
                f"Decision: {decision}",
                f"Reason: {reason}",
                f"Best role: {best_role}",
                f"Best score: {float(best_score):.3f}" if best_score is not None else "Best score: —",
            ]
        )
    )

    out.append("\nMatched Entity")
    out.append("-------------")
    if isinstance(best_wl, Mapping) and best_wl:
        out.append(
            f"{best_wl.get('name','—')} (List: {best_wl.get('list','—')}; Category: {best_wl.get('category','—')}; Country: {best_wl.get('country','—')}; DOB: {best_wl.get('dob','—')})"
        )
    else:
        out.append("—")

    out.append("\nKey Drivers")
    out.append("-----------")
    if drivers:
        for k, v in drivers:
            out.append(f"- {k.capitalize()}: {v:.3f}")
    else:
        out.append("- —")

    out.append("\nSanctions")
    out.append("---------")
    if sanction_flag:
        out.append("Sanctions hit detected:")
        for r in sanction_reasons:
            out.append(f"- {r}")
    else:
        out.append("No sanctions hit detected.")

    out.append("\nAutomated Decisioning")
    out.append("---------------------")
    if decision == "ESCALATE":
        out.append("- Place payment on hold and route to Level-2 review.")
        out.append("- Validate KYC/KYB and re-screen against freshest lists.")
        out.append("- If confirmed sanctions, follow blocking/reporting SOP.")
    else:
        out.append("- Proceed with STP release per policy.")
        out.append("- Retain logs, scores and evidence for audit.")

    out.append("\nEvidence Snapshot")
    out.append("-----------------")
    out.append(
        f"PAYER: {payload.get('payer_name','')} | {payload.get('payer_country','')} | {payload.get('payer_address','')}"
    )
    out.append(
        f"BENEFICIARY: {payload.get('benef_name','')} | {payload.get('benef_country','')} | {payload.get('benef_address','')}"
    )

    return "\n".join(out)


def _maybe_llm_rewrite_explanation(deterministic_explanation: str) -> Optional[str]:
    """Optional LLM rewrite. Safe-by-default: returns None if not configured.

    Env vars (OpenAI-compatible):
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default: https://api.openai.com)
    - OPENAI_MODEL (default: gpt-4o-mini)

    This is deliberately best-effort and will never throw.
    """

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

    try:
        import urllib.request

        url = f"{base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a compliance analyst. Rewrite the provided payment screening explanation "
                        "to be concise, audit-ready, and grounded only in the provided text. Do not invent facts."
                    ),
                },
                {"role": "user", "content": deterministic_explanation},
            ],
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content") or "").strip()
        return content or None
    except Exception:
        return None


def build_payment_screening_agent(*, audit_path: Optional[str] = None) -> Agent:
    def step_sample(ctx: Context) -> StepResult:
        ctx.audit("sample_data", "Received input payload", {"keys": sorted(list(ctx.input.keys()))})
        coerced = core.coerce_payload(ctx.input)
        return StepResult.ok({"payload": coerced})

    def step_validate(ctx: Context) -> StepResult:
        payload = ctx.state.get("payload") or {}
        ok, missing = core.validate_payload(payload)
        if not ok:
            ctx.audit("validate", "Payload missing required fields", {"missing": missing})
            return StepResult.fail(f"Missing required fields: {missing}")
        ctx.audit(
            "validate",
            "Validated payload",
            {"required_fields": list(core.REQUIRED_FIELDS)},
        )
        return StepResult.ok()

    def step_match(ctx: Context) -> StepResult:
        payload = ctx.state.get("payload") or {}
        ctx.audit("match", "Matching against watchlist", {"watchlist_size": len(core.WATCHLIST)})
        candidates, sanction_flag_any, sanction_reasons = core.build_candidates(payload)
        ctx.audit(
            "match",
            "Generated candidate matches",
            {"candidate_count": len(candidates), "sanction_flag": sanction_flag_any},
        )
        # Convert to JSON-friendly dicts
        candidates_out = [
            {
                "role": c.role,
                "wl": c.watchlist,
                "score": float(c.score),
                "breakdown": c.breakdown,
            }
            for c in sorted(candidates, key=lambda x: x.score, reverse=True)
        ]
        best = core.pick_best_candidate(candidates)
        return StepResult.ok(
            {
                "candidates": candidates_out,
                "best_role": best.role if best else None,
                "best_wl": best.watchlist if best else None,
                "best_score": float(best.score) if best else 0.0,
                "breakdown": best.breakdown if best else {},
                "sanction_flag": sanction_flag_any,
                "sanction_reasons": sanction_reasons,
            }
        )

    def step_decide(ctx: Context) -> StepResult:
        best_score = float(ctx.state.get("best_score") or 0.0)
        sanction_flag_any = bool(ctx.state.get("sanction_flag") or False)
        decision, reason = core.apply_decision_policy(best_score, sanction_flag_any)
        ctx.audit(
            "decision",
            "Applied automated decision policy",
            {
                "decision": decision,
                "reason": reason,
                "threshold": core.THRESHOLD,
                "best_score": best_score,
                "sanction_flag": sanction_flag_any,
            },
        )
        return StepResult.ok({"decision": decision, "reason": reason})

    def step_explain(ctx: Context) -> StepResult:
        payload = ctx.state.get("payload") or {}
        result_view = {
            "decision": ctx.state.get("decision"),
            "reason": ctx.state.get("reason"),
            "best_role": ctx.state.get("best_role"),
            "best_wl": ctx.state.get("best_wl"),
            "best_score": ctx.state.get("best_score"),
            "breakdown": ctx.state.get("breakdown"),
            "sanction_flag": ctx.state.get("sanction_flag"),
            "sanction_reasons": ctx.state.get("sanction_reasons"),
        }
        deterministic = generate_explanation(payload, result_view)

        use_llm = bool(ctx.input.get("use_llm_explainer"))
        if use_llm:
            llm = _maybe_llm_rewrite_explanation(deterministic)
            if llm:
                ctx.audit("explain", "Generated LLM explanation", {"chars": len(llm)})
                return StepResult.ok({"explanation": llm, "explanation_mode": "llm"})
            ctx.audit("explain", "LLM not configured; using deterministic explanation", {})

        ctx.audit("explain", "Generated deterministic explanation", {"chars": len(deterministic)})
        return StepResult.ok({"explanation": deterministic, "explanation_mode": "deterministic"})

    flow = Flow(
        name="payment_screening_agent",
        steps=[
            Step(name="sample_data", fn=step_sample),
            Step(name="validate", fn=step_validate),
            Step(name="match", fn=step_match),
            Step(name="decision", fn=step_decide),
            Step(name="explain", fn=step_explain),
        ],
    )

    audit_sink = JsonlAuditSink(audit_path or _default_audit_path())
    return Agent(flow=flow, audit_sink=audit_sink)


def screen_payment_agentic(
    payload: Mapping[str, Any],
    *,
    job_id: Optional[str] = None,
    item_index: int = 0,
    audit_events: Optional[List[Dict[str, Any]]] = None,
    audit_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the agentic screening flow.

    Returns a result dict suitable for UI/API use.

    - If audit_events is provided, events will be appended as dicts.
    - Audit is always appended (best-effort) to audit_trail.jsonl.
    """

    job_id = job_id or uuid.uuid4().hex
    audit_events = audit_events if audit_events is not None else []

    jsonl = JsonlAuditSink(audit_path or _default_audit_path())
    mem = DictListAuditSink(audit_events)

    # Fresh agent instance per call (small + avoids shared state).
    agent = build_payment_screening_agent(audit_path=audit_path or _default_audit_path())

    # Override audit sink to include list sink as well.
    agent.audit_sink = CompositeAuditSink([jsonl, mem])

    res = agent.run(dict(payload), job_id=job_id, item_index=item_index)
    if not res.ok:
        return {
            "decision": "ERROR",
            "reason": res.error or "Unknown error",
            "best_role": None,
            "best_wl": None,
            "best_score": float(res.final_state.get("best_score") or 0.0),
            "breakdown": res.final_state.get("breakdown") or {},
            "sanction_flag": bool(res.final_state.get("sanction_flag") or False),
            "sanction_reasons": res.final_state.get("sanction_reasons") or [],
            "candidates": res.final_state.get("candidates") or [],
            "explanation": res.final_state.get("explanation") or ("Flow failed: " + (res.error or "")),
            "explanation_mode": res.final_state.get("explanation_mode") or "deterministic",
            "job_id": job_id,
        }

    return {
        "decision": res.final_state.get("decision"),
        "reason": res.final_state.get("reason"),
        "best_role": res.final_state.get("best_role"),
        "best_wl": res.final_state.get("best_wl"),
        "best_score": float(res.final_state.get("best_score") or 0.0),
        "breakdown": res.final_state.get("breakdown") or {},
        "sanction_flag": bool(res.final_state.get("sanction_flag") or False),
        "sanction_reasons": res.final_state.get("sanction_reasons") or [],
        "candidates": res.final_state.get("candidates") or [],
        "explanation": res.final_state.get("explanation") or "",
        "explanation_mode": res.final_state.get("explanation_mode") or "deterministic",
        "job_id": job_id,
    }

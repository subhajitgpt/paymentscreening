from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

app = Flask(__name__)

# -----------------------------
# Demo watchlist / sanctions
# -----------------------------
WATCHLIST = [
    {
        "name": "ABC",
        "aka": ["sanction hit", "sanction hit"],
        "address": "sanction hit, sanction hit",
        "country": "sanction hit",
        "dob": "1990-02-01",
        "list": "Consolidated",
        "category": "Corruption",
    },
    {
        "name": "XYZ",
        "aka": ["Global Trading Limited", "Global Trade Co."],
        "address": "xyzcity",
        "country": "xyzcity",
        "dob": None,
        "list": "Internal Watch",
        "category": "Adverse Media",
    },
]

SANCTIONED_COUNTRIES_RAW = ["pakistan", "iran", "syria", "ukraine", "cuba", "north korea"]
SANCTION_ALIASES = {"ukraise": "ukraine", "u k r a i s e": "ukraine"}


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\.,;:\-\(\)\[\]/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_country(s: str) -> str:
    s = _norm(s)
    if not s:
        return ""
    return SANCTION_ALIASES.get(s, s)


SANCTIONED_COUNTRIES = set(canonical_country(c) for c in SANCTIONED_COUNTRIES_RAW)


def address_has_sanctioned_country(addr: str) -> Tuple[bool, Optional[str]]:
    text = _norm(addr)
    for sc in SANCTIONED_COUNTRIES:
        if re.search(rf"\b{re.escape(sc)}\b", text):
            return True, sc
    return False, None


def is_sanctioned_country(country_str: str) -> Tuple[bool, str]:
    can = canonical_country(country_str)
    return (can in SANCTIONED_COUNTRIES), can


# -----------------------------
# Text normalization & similarity
# -----------------------------
ABBR = {
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bp\.?\s*o\.?\s*box\b": "po box",
}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[\(\)\[\]\.,;:!@#\$%\^&\*\-_/\\]+", " ", s)
    for pat, repl in ABBR.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"\s+", normalize_text(s)) if t]


def jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def jaro_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1 or "", s2 or ""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    return jaro


def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    s1n, s2n = normalize_text(s1), normalize_text(s2)
    j = jaro_similarity(s1n, s2n)
    l = 0
    for a, b in zip(s1n, s2n):
        if a == b:
            l += 1
            if l == max_l:
                break
        else:
            break
    return j + l * p * (1 - j)


# -----------------------------
# Scoring and decision policy
# -----------------------------
THRESHOLD = 0.80


def name_similarity(input_name: str, wl_name: str, wl_aka: Optional[List[str]]) -> float:
    candidates = [wl_name] + (wl_aka or [])
    sims: List[float] = []
    for cand in candidates:
        sims.append(jaro_winkler(input_name, cand))
        sims.append(0.5 * jaccard(tokenize(input_name), tokenize(cand)))
    return max(sims) if sims else 0.0


def address_similarity(input_addr: str, wl_addr: str) -> float:
    jw = jaro_winkler(input_addr, wl_addr)
    jac = jaccard(tokenize(input_addr), tokenize(wl_addr))
    return 0.4 * jw + 0.6 * jac


def dob_similarity(input_dob: str, wl_dob: Optional[str]) -> float:
    if not input_dob or not wl_dob:
        return 0.0
    try:
        d1 = datetime.strptime(input_dob, "%Y-%m-%d").date()
        d2 = datetime.strptime(wl_dob, "%Y-%m-%d").date()
        return 1.0 if d1 == d2 else 0.0
    except Exception:
        return 0.0


def country_bonus(input_country: str, wl_country: Optional[str]) -> float:
    if not input_country or not wl_country:
        return 0.0
    return 0.05 if _norm(input_country).upper() == _norm(wl_country).upper() else 0.0


def composite_risk_score(
    name_in: str,
    addr_in: str,
    country_in: str,
    dob_in: str,
    wl: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    n = name_similarity(name_in, wl["name"], wl.get("aka"))
    a = address_similarity(addr_in, wl["address"])
    d = dob_similarity(dob_in, wl.get("dob"))
    c = country_bonus(country_in, wl.get("country"))

    score = 0.60 * n + 0.30 * a + 0.05 * d + c
    score = min(1.0, score)

    party_sanction_hit, party_sanction_can = is_sanctioned_country(country_in)
    addr_hit, addr_match = address_has_sanctioned_country(addr_in)

    return score, {
        "name": n,
        "address": a,
        "dob": d,
        "country": c,
        "sanction_party_country": party_sanction_hit,
        "sanction_party_country_name": party_sanction_can if party_sanction_hit else None,
        "sanction_address_hit": addr_hit,
        "sanction_address_match": addr_match,
    }


def _validate_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    required = [
        "payer_name",
        "payer_address",
        "payer_country",
        "payer_dob",
        "benef_name",
        "benef_address",
        "benef_country",
        "benef_dob",
    ]
    missing = [k for k in required if k not in payload]
    return (len(missing) == 0), missing


def _coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for k in [
        "payer_name",
        "payer_address",
        "payer_country",
        "payer_dob",
        "benef_name",
        "benef_address",
        "benef_country",
        "benef_dob",
        "currency",
        "reference",
    ]:
        if k in out and out[k] is None:
            out[k] = ""
        if k in out and not isinstance(out[k], str):
            out[k] = str(out[k])

    if "amount" in out:
        try:
            out["amount"] = float(out["amount"])
        except Exception:
            pass

    for k in ["payer_dob", "benef_dob"]:
        out.setdefault(k, "")

    return out


# -----------------------------
# Agentic flow (traceable)
# -----------------------------
@dataclass
class AuditEvent:
    ts: str
    job_id: str
    item_index: Optional[int]
    step: str
    message: str
    data: Dict[str, Any]


AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), "audit_trail.jsonl")


def _append_audit(event: AuditEvent) -> None:
    line = {
        "ts": event.ts,
        "job_id": event.job_id,
        "item_index": event.item_index,
        "step": event.step,
        "message": event.message,
        "data": event.data,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        # Best effort: still keep in-memory audit.
        pass


def generate_explanation(payload: Dict[str, Any], result: Dict[str, Any]) -> str:
    decision = result.get("decision")
    reason = result.get("reason")
    best_score = result.get("best_score")
    best_role = result.get("best_role")
    best_wl = result.get("best_wl") or {}
    breakdown = result.get("breakdown") or {}
    sanction_flag = result.get("sanction_flag") or False
    sanction_reasons = result.get("sanction_reasons") or []

    drivers = []
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
    if best_wl:
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


def run_agentic_screening(
    payload: Dict[str, Any],
    *,
    job_id: str,
    item_index: Optional[int],
    audit_sink: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def audit(step: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        ev = AuditEvent(
            ts=_utc_now_iso(),
            job_id=job_id,
            item_index=item_index,
            step=step,
            message=message,
            data=data or {},
        )
        audit_sink.append(
            {
                "ts": ev.ts,
                "job_id": ev.job_id,
                "item_index": ev.item_index,
                "step": ev.step,
                "message": ev.message,
                "data": ev.data,
            }
        )
        _append_audit(ev)

    audit("sample_data", "Received input payload", {"keys": sorted(list(payload.keys()))})

    payload = _coerce_payload(payload)
    ok, missing = _validate_payload(payload)
    if not ok:
        audit("validate", "Payload missing required fields", {"missing": missing})
        raise ValueError(f"Missing required fields: {missing}")

    audit(
        "algorithms",
        "Prepared normalization + similarity algorithms",
        {"algorithms": ["normalize_text", "Jaro-Winkler", "Jaccard"]},
    )

    # Score all payer/beneficiary candidates
    audit("match", "Matching against watchlist", {"watchlist_size": len(WATCHLIST)})

    candidates = []
    sanction_flag_any = False
    sanction_reasons: List[str] = []

    for wl in WATCHLIST:
        s_payer, bd_payer = composite_risk_score(
            payload["payer_name"],
            payload["payer_address"],
            payload["payer_country"],
            payload.get("payer_dob", ""),
            wl,
        )
        candidates.append(("PAYER", wl, s_payer, bd_payer))

        s_benef, bd_benef = composite_risk_score(
            payload["benef_name"],
            payload["benef_address"],
            payload["benef_country"],
            payload.get("benef_dob", ""),
            wl,
        )
        candidates.append(("BENEFICIARY", wl, s_benef, bd_benef))

        for bd, role in [(bd_payer, "PAYER"), (bd_benef, "BENEFICIARY")]:
            if bd.get("sanction_party_country"):
                sanction_flag_any = True
                sanction_reasons.append(
                    f"{role} country in sanctioned list: {bd.get('sanction_party_country_name')}"
                )
            if bd.get("sanction_address_hit"):
                sanction_flag_any = True
                sanction_reasons.append(
                    f"{role} address mentions sanctioned country: {bd.get('sanction_address_match')}"
                )

    best = max(candidates, key=lambda x: x[2]) if candidates else None
    best_role, best_wl, best_score, best_breakdown = best if best else (None, None, 0.0, {})

    audit(
        "scoring",
        "Computed composite risk score",
        {
            "threshold": THRESHOLD,
            "best_role": best_role,
            "best_score": float(best_score or 0.0),
            "breakdown": best_breakdown,
        },
    )

    if sanction_flag_any:
        decision = "ESCALATE"
        reason = "Sanctioned Country"
    else:
        decision = "ESCALATE" if float(best_score or 0.0) >= THRESHOLD else "RELEASE"
        reason = "Score Threshold" if decision == "ESCALATE" else "Below Threshold"

    audit(
        "decision",
        "Applied automated decision policy",
        {
            "decision": decision,
            "reason": reason,
            "sanction_flag": sanction_flag_any,
            "sanction_reasons": sanction_reasons,
        },
    )

    sorted_cands = sorted(
        [
            {"role": r, "wl": w, "score": s, "breakdown": bd}
            for r, w, s, bd in candidates
            if w is not None
        ],
        key=lambda z: z["score"],
        reverse=True,
    )

    result = {
        "decision": decision,
        "reason": reason,
        "best_role": best_role,
        "best_wl": best_wl,
        "best_score": float(best_score or 0.0),
        "breakdown": best_breakdown,
        "sanction_flag": sanction_flag_any,
        "sanction_reasons": sanction_reasons,
        "candidates": sorted_cands,
    }

    explanation = generate_explanation(payload, result)
    result["explanation"] = explanation

    audit("explain", "Generated compliance explanation", {"chars": len(explanation)})
    return result


# -----------------------------
# Batch jobs + KPIs
# -----------------------------
EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def _new_job(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = _utc_now_iso()
    return {
        "job_id": job_id,
        "status": "QUEUED",  # QUEUED|RUNNING|COMPLETED|FAILED
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "total_items": len(items),
        "completed_items": 0,
        "escalate_count": 0,
        "release_count": 0,
        "max_score": 0.0,
        "avg_score": 0.0,
        "items": items,
        "results": [],
        "audit": [],
        "error": None,
    }


def _kpis_snapshot() -> Dict[str, Any]:
    with JOBS_LOCK:
        jobs = list(JOBS.values())

    total = len(jobs)
    running = sum(1 for j in jobs if j["status"] == "RUNNING")
    queued = sum(1 for j in jobs if j["status"] == "QUEUED")
    completed = sum(1 for j in jobs if j["status"] == "COMPLETED")
    failed = sum(1 for j in jobs if j["status"] == "FAILED")

    total_items = sum(int(j.get("total_items") or 0) for j in jobs)
    done_items = sum(int(j.get("completed_items") or 0) for j in jobs)

    escalated = sum(int(j.get("escalate_count") or 0) for j in jobs)
    released = sum(int(j.get("release_count") or 0) for j in jobs)

    decision_total = escalated + released
    escalate_rate = (escalated / decision_total) if decision_total else 0.0

    return {
        "jobs_total": total,
        "jobs_queued": queued,
        "jobs_running": running,
        "jobs_completed": completed,
        "jobs_failed": failed,
        "items_total": total_items,
        "items_completed": done_items,
        "decisions_escalate": escalated,
        "decisions_release": released,
        "escalate_rate": escalate_rate,
    }


def _process_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "RUNNING"
        job["started_at"] = _utc_now_iso()

    start = time.time()
    try:
        items = job["items"]
        for idx, payload in enumerate(items):
            # Record a job-level audit entry per item
            result = run_agentic_screening(
                payload,
                job_id=job_id,
                item_index=idx,
                audit_sink=job["audit"],
            )

            with JOBS_LOCK:
                job["results"].append({"index": idx, "payload": payload, "result": result})
                job["completed_items"] += 1
                if result.get("decision") == "ESCALATE":
                    job["escalate_count"] += 1
                else:
                    job["release_count"] += 1

                score = float(result.get("best_score") or 0.0)
                job["max_score"] = max(float(job.get("max_score") or 0.0), score)

                # Streaming average
                n = int(job.get("completed_items") or 1)
                prev_avg = float(job.get("avg_score") or 0.0)
                job["avg_score"] = prev_avg + (score - prev_avg) / n

        with JOBS_LOCK:
            job["status"] = "COMPLETED"
            job["finished_at"] = _utc_now_iso()

            duration_s = max(0.0, time.time() - start)
            job["audit"].append(
                {
                    "ts": _utc_now_iso(),
                    "job_id": job_id,
                    "item_index": None,
                    "step": "job_complete",
                    "message": "Job completed",
                    "data": {"duration_s": duration_s},
                }
            )

    except Exception as e:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["status"] = "FAILED"
                job["finished_at"] = _utc_now_iso()
                job["error"] = str(e)
                job["audit"].append(
                    {
                        "ts": _utc_now_iso(),
                        "job_id": job_id,
                        "item_index": None,
                        "step": "job_failed",
                        "message": "Job failed",
                        "data": {"error": str(e)},
                    }
                )


def _parse_batch_json(raw: bytes) -> List[Dict[str, Any]]:
    obj = json.loads(raw.decode("utf-8"))

    # Accept:
    # 1) list[payload]
    # 2) {"items": [payload,...]}
    # 3) same shape as test_payloads.json: {"test_scenarios": {name: {payload: {...}}}}
    if isinstance(obj, list):
        return [dict(x) for x in obj]

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return [dict(x) for x in obj["items"]]

    if isinstance(obj, dict) and isinstance(obj.get("test_scenarios"), dict):
        items: List[Dict[str, Any]] = []
        for _, entry in obj["test_scenarios"].items():
            if isinstance(entry, dict) and isinstance(entry.get("payload"), dict):
                items.append(dict(entry["payload"]))
        return items

    raise ValueError("Unsupported batch JSON format. Expected list, {items:[...]}, or test_scenarios format.")


def _load_test_scenarios() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "test_payloads.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenarios = data.get("test_scenarios") or {}
        out = {}
        for name, entry in scenarios.items():
            if isinstance(entry, dict) and isinstance(entry.get("payload"), dict):
                out[name] = entry["payload"]
        return out
    except Exception:
        return {}


# -----------------------------
# UI
# -----------------------------
TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Payment Screening v4 (Agentic)</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <style>
    body { background: #0f172a; color: #e2e8f0; }
    .card { border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,.3); }
    .muted { color: #94a3b8; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
<div class=\"container py-4\">
  <div class=\"card border-0\">
    <div class=\"card-header bg-dark\">
      <div class=\"d-flex justify-content-between align-items-start\">
        <div>
          <h3 class=\"m-0\">Payment Screening v4 (Agentic Flow)</h3>
          <div class=\"muted\">Sample data → Algorithms → Scoring → Automated decision → Explanation • Batch jobs + audit trail</div>
        </div>
        <div class=\"text-end\">
          <a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('index') }}\">Refresh</a>
        </div>
      </div>
    </div>

    <div class=\"card-body bg-dark\">
      <ul class=\"nav nav-tabs\" role=\"tablist\">
        <li class=\"nav-item\" role=\"presentation\">
          <button class=\"nav-link active\" data-bs-toggle=\"tab\" data-bs-target=\"#tab-main\" type=\"button\" role=\"tab\">Main (KPIs + Batch)</button>
        </li>
        <li class=\"nav-item\" role=\"presentation\">
          <button class=\"nav-link\" data-bs-toggle=\"tab\" data-bs-target=\"#tab-jobs\" type=\"button\" role=\"tab\">Jobs + Audit Trail</button>
        </li>
      </ul>

      <div class=\"tab-content pt-3\">
        <!-- MAIN TAB -->
        <div class=\"tab-pane fade show active\" id=\"tab-main\" role=\"tabpanel\">
          <div class=\"row g-3\">
            <div class=\"col-md-3\">
              <div class=\"border rounded p-3\">
                <div class=\"muted\">Jobs</div>
                <div class=\"fs-4\">{{ kpis.jobs_total }}</div>
                <div class=\"muted\">Queued: {{ kpis.jobs_queued }} • Running: {{ kpis.jobs_running }} • Failed: {{ kpis.jobs_failed }}</div>
              </div>
            </div>
            <div class=\"col-md-3\">
              <div class=\"border rounded p-3\">
                <div class=\"muted\">Items Processed</div>
                <div class=\"fs-4\">{{ kpis.items_completed }} / {{ kpis.items_total }}</div>
                <div class=\"muted\">Across all batch jobs</div>
              </div>
            </div>
            <div class=\"col-md-3\">
              <div class=\"border rounded p-3\">
                <div class=\"muted\">Decisions</div>
                <div class=\"fs-4\">E: {{ kpis.decisions_escalate }} • R: {{ kpis.decisions_release }}</div>
                <div class=\"muted\">Escalate rate: {{ (kpis.escalate_rate * 100) | round(1) }}%</div>
              </div>
            </div>
            <div class=\"col-md-3\">
              <div class=\"border rounded p-3\">
                <div class=\"muted\">Policy</div>
                <div class=\"fs-4\">Threshold: {{ threshold }}</div>
                <div class=\"muted\">Hard rule: sanctioned country/address → ESCALATE</div>
              </div>
            </div>
          </div>

          <hr class=\"border-secondary\" />

          <div class=\"row g-3\">
            <div class=\"col-lg-6\">
              <h5 class=\"text-info\">Single Transaction (Sync)</h5>

              <form method=\"get\" class=\"mb-3\">
                <label class=\"form-label\">Load sample scenario</label>
                <div class=\"input-group\">
                  <select class=\"form-select\" name=\"scenario\">
                    <option value=\"\">-- Choose --</option>
                    {% for s in scenarios %}
                      <option value=\"{{ s }}\" {% if s == selected_scenario %}selected{% endif %}>{{ s }}</option>
                    {% endfor %}
                  </select>
                  <button class=\"btn btn-outline-light\" type=\"submit\">Load</button>
                </div>
                <div class=\"muted mt-1\">Uses data from <span class=\"mono\">test_payloads.json</span></div>
              </form>

              <form method=\"post\" action=\"{{ url_for('screen_sync') }}\" class=\"row g-2\">
                <div class=\"col-12\"><div class=\"muted\">Payer</div></div>
                <div class=\"col-md-6\"><input class=\"form-control\" name=\"payer_name\" placeholder=\"payer_name\" value=\"{{ form.payer_name }}\" required></div>
                <div class=\"col-md-6\"><input class=\"form-control\" name=\"payer_address\" placeholder=\"payer_address\" value=\"{{ form.payer_address }}\" required></div>
                <div class=\"col-md-4\"><input class=\"form-control\" name=\"payer_country\" placeholder=\"payer_country\" value=\"{{ form.payer_country }}\" required></div>
                <div class=\"col-md-8\"><input class=\"form-control\" name=\"payer_dob\" placeholder=\"payer_dob (YYYY-MM-DD)\" value=\"{{ form.payer_dob }}\"></div>

                <div class=\"col-12 pt-2\"><div class=\"muted\">Beneficiary</div></div>
                <div class=\"col-md-6\"><input class=\"form-control\" name=\"benef_name\" placeholder=\"benef_name\" value=\"{{ form.benef_name }}\" required></div>
                <div class=\"col-md-6\"><input class=\"form-control\" name=\"benef_address\" placeholder=\"benef_address\" value=\"{{ form.benef_address }}\" required></div>
                <div class=\"col-md-4\"><input class=\"form-control\" name=\"benef_country\" placeholder=\"benef_country\" value=\"{{ form.benef_country }}\" required></div>
                <div class=\"col-md-8\"><input class=\"form-control\" name=\"benef_dob\" placeholder=\"benef_dob (YYYY-MM-DD)\" value=\"{{ form.benef_dob }}\"></div>

                <div class=\"col-md-4\"><input class=\"form-control\" name=\"amount\" placeholder=\"amount\" value=\"{{ form.amount }}\"></div>
                <div class=\"col-md-4\"><input class=\"form-control\" name=\"currency\" placeholder=\"currency\" value=\"{{ form.currency }}\"></div>
                <div class=\"col-md-4\"><input class=\"form-control\" name=\"reference\" placeholder=\"reference\" value=\"{{ form.reference }}\"></div>

                <div class=\"col-12 pt-2\">
                  <button class=\"btn btn-info\" type=\"submit\">Run Agentic Screening</button>
                </div>
              </form>

              {% if sync_result %}
                <hr class=\"border-secondary\" />
                <h6 class=\"text-warning\">Result</h6>
                <div class=\"border rounded p-3\">
                  <div><b>Decision:</b> {{ sync_result.decision }} ({{ sync_result.reason }})</div>
                  <div><b>Best:</b> {{ sync_result.best_role }} • Score {{ sync_result.best_score | round(3) }}</div>
                  <div class=\"muted\">Sanction flag: {{ sync_result.sanction_flag }}</div>
                </div>
                <div class=\"mt-3\">
                  <h6 class=\"text-warning\">Explanation</h6>
                  <pre class=\"border rounded p-3 bg-black\">{{ sync_result.explanation }}</pre>
                </div>
              {% endif %}
            </div>

            <div class=\"col-lg-6\">
              <h5 class=\"text-info\">Batch Screening (Async Job)</h5>
              <div class=\"muted\">Upload JSON as: list[payload] OR {items:[...]} OR same shape as <span class=\"mono\">test_payloads.json</span>.</div>

              <form method=\"post\" action=\"{{ url_for('submit_job') }}\" enctype=\"multipart/form-data\" class=\"mt-2\">
                <div class=\"mb-2\">
                  <input class=\"form-control\" type=\"file\" name=\"batch_file\" accept=\"application/json\" required>
                </div>
                <button class=\"btn btn-outline-info\" type=\"submit\">Submit Batch Job</button>
              </form>

              {% if submit_msg %}
                <div class=\"alert alert-secondary mt-3\">{{ submit_msg }}</div>
              {% endif %}

              <hr class=\"border-secondary\" />
              <h6 class=\"text-warning\">Recent Jobs</h6>
              <div class=\"table-responsive\">
                <table class=\"table table-dark table-sm align-middle\">
                  <thead>
                    <tr>
                      <th>Job</th>
                      <th>Status</th>
                      <th>Done</th>
                      <th>E/R</th>
                      <th>Avg</th>
                      <th>Max</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {% for j in jobs %}
                      <tr>
                        <td class=\"mono\">{{ j.job_id[:8] }}</td>
                        <td>{{ j.status }}</td>
                        <td>{{ j.completed_items }}/{{ j.total_items }}</td>
                        <td>{{ j.escalate_count }}/{{ j.release_count }}</td>
                        <td>{{ j.avg_score | round(3) }}</td>
                        <td>{{ j.max_score | round(3) }}</td>
                        <td><a class=\"btn btn-sm btn-outline-light\" href=\"{{ url_for('job_view', job_id=j.job_id) }}\">Open</a></td>
                      </tr>
                    {% endfor %}
                    {% if not jobs %}
                      <tr><td colspan=\"7\" class=\"muted\">No jobs yet.</td></tr>
                    {% endif %}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <!-- JOBS TAB -->
        <div class=\"tab-pane fade\" id=\"tab-jobs\" role=\"tabpanel\">
          <h5 class=\"text-info\">Job Status & Audit Trail</h5>
          <div class=\"muted\">Audit events are also persisted to <span class=\"mono\">audit_trail.jsonl</span> (best-effort).</div>

          <div class=\"mt-3\">
            <div class=\"table-responsive\">
              <table class=\"table table-dark table-sm align-middle\">
                <thead>
                  <tr>
                    <th>Job</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Started</th>
                    <th>Finished</th>
                    <th>Done</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {% for j in jobs %}
                    <tr>
                      <td class=\"mono\">{{ j.job_id }}</td>
                      <td>{{ j.status }}</td>
                      <td class=\"mono\">{{ j.created_at }}</td>
                      <td class=\"mono\">{{ j.started_at or '—' }}</td>
                      <td class=\"mono\">{{ j.finished_at or '—' }}</td>
                      <td>{{ j.completed_items }}/{{ j.total_items }}</td>
                      <td><a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('job_view', job_id=j.job_id) }}\">View + Audit</a></td>
                    </tr>
                  {% endfor %}
                  {% if not jobs %}
                    <tr><td colspan=\"7\" class=\"muted\">No jobs yet.</td></tr>
                  {% endif %}
                </tbody>
              </table>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <div class=\"muted mt-3\">
    API: <span class=\"mono\">/api/jobs</span>, <span class=\"mono\">/api/jobs/&lt;job_id&gt;</span>, <span class=\"mono\">/api/jobs/&lt;job_id&gt;/audit</span>
  </div>
</div>

<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body>
</html>
"""


JOB_TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Job {{ job.job_id }}</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <style>
    body { background: #0f172a; color: #e2e8f0; }
    .muted { color: #94a3b8; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
<div class=\"container py-4\">
  <div class=\"d-flex justify-content-between align-items-start\">
    <div>
      <h3 class=\"m-0\">Job</h3>
      <div class=\"mono\">{{ job.job_id }}</div>
      <div class=\"muted\">Status: {{ job.status }} • Done {{ job.completed_items }}/{{ job.total_items }}</div>
    </div>
    <div class=\"text-end\">
      <a class=\"btn btn-sm btn-outline-light\" href=\"{{ url_for('index') }}\">Back</a>
      <a class=\"btn btn-sm btn-outline-info\" href=\"{{ url_for('job_view', job_id=job.job_id) }}\">Refresh</a>
    </div>
  </div>

  {% if job.error %}
    <div class=\"alert alert-danger mt-3\">{{ job.error }}</div>
  {% endif %}

  <hr class=\"border-secondary\" />
  <h5 class=\"text-info\">Results</h5>
  <div class=\"table-responsive\">
    <table class=\"table table-dark table-sm align-middle\">
      <thead>
        <tr>
          <th>#</th>
          <th>Decision</th>
          <th>Reason</th>
          <th>Best Role</th>
          <th>Best Score</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {% for r in job.results %}
          <tr>
            <td>{{ r.index }}</td>
            <td>{{ r.result.decision }}</td>
            <td>{{ r.result.reason }}</td>
            <td>{{ r.result.best_role }}</td>
            <td>{{ r.result.best_score | round(3) }}</td>
            <td>
              <details>
                <summary class=\"text-warning\">Explanation</summary>
                <pre class=\"border rounded p-2 bg-black\">{{ r.result.explanation }}</pre>
              </details>
            </td>
          </tr>
        {% endfor %}
        {% if not job.results %}
          <tr><td colspan=\"6\" class=\"muted\">No results yet.</td></tr>
        {% endif %}
      </tbody>
    </table>
  </div>

  <hr class=\"border-secondary\" />
  <h5 class=\"text-info\">Audit Trail (In-Memory)</h5>
  <div class=\"muted\">Also appended to <span class=\"mono\">audit_trail.jsonl</span>.</div>
  <pre class=\"border rounded p-3 bg-black mt-2\">{% for a in job.audit %}{{ a.ts }}  [{{ a.step }}]  item={{ a.item_index if a.item_index is not none else '-' }}  {{ a.message }}\n{% endfor %}{% if not job.audit %}—{% endif %}</pre>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    scenarios_map = _load_test_scenarios()
    selected_scenario = (request.args.get("scenario") or "").strip()
    payload = scenarios_map.get(selected_scenario) if selected_scenario else None

    form = _coerce_payload(
        payload
        or {
            "payer_name": "Global Trade LLC",
            "payer_address": "PO Box 12345, Dubai, UAE",
            "payer_country": "AE",
            "payer_dob": "",
            "benef_name": "Olena Petrenko-Kovalenko",
            "benef_address": "Kreschatyk 22, Kyiv, Ukraine",
            "benef_country": "UA",
            "benef_dob": "1990-02-01",
            "amount": 12500.0,
            "currency": "USD",
            "reference": "Invoice 2025-10-ACME",
        }
    )

    with JOBS_LOCK:
        jobs = list(JOBS.values())
    jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

    return render_template_string(
        TEMPLATE,
        kpis=_kpis_snapshot(),
        threshold=THRESHOLD,
        scenarios=sorted(list(scenarios_map.keys())),
        selected_scenario=selected_scenario,
        form=form,
        sync_result=None,
        submit_msg=None,
        jobs=jobs,
    )


@app.route("/screen/sync", methods=["POST"])
def screen_sync():
    form = _coerce_payload(dict(request.form))
    job_id = "sync-" + uuid.uuid4().hex
    audit_sink: List[Dict[str, Any]] = []

    try:
        result = run_agentic_screening(form, job_id=job_id, item_index=0, audit_sink=audit_sink)
    except Exception as e:
        result = {
            "decision": "ERROR",
            "reason": str(e),
            "best_role": None,
            "best_wl": None,
            "best_score": 0.0,
            "breakdown": {},
            "sanction_flag": False,
            "sanction_reasons": [],
            "candidates": [],
            "explanation": f"Failed to screen payload: {e}",
        }

    scenarios_map = _load_test_scenarios()
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

    return render_template_string(
        TEMPLATE,
        kpis=_kpis_snapshot(),
        threshold=THRESHOLD,
        scenarios=sorted(list(scenarios_map.keys())),
        selected_scenario="",
        form=form,
        sync_result=result,
        submit_msg=None,
        jobs=jobs,
    )


@app.route("/jobs/submit", methods=["POST"])
def submit_job():
    f = request.files.get("batch_file")
    if not f:
        return redirect(url_for("index"))

    try:
        items = _parse_batch_json(f.read())
        if not items:
            raise ValueError("No payloads found in uploaded JSON")

        job = _new_job(items)
        with JOBS_LOCK:
            JOBS[job["job_id"]] = job

        EXECUTOR.submit(_process_job, job["job_id"])
        msg = f"Submitted job {job['job_id']} with {job['total_items']} item(s)."

    except Exception as e:
        msg = f"Batch upload failed: {e}"

    scenarios_map = _load_test_scenarios()
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    jobs = sorted(jobs, key=lambda j: j.get("created_at") or "", reverse=True)[:20]

    return render_template_string(
        TEMPLATE,
        kpis=_kpis_snapshot(),
        threshold=THRESHOLD,
        scenarios=sorted(list(scenarios_map.keys())),
        selected_scenario="",
        form=_coerce_payload({}),
        sync_result=None,
        submit_msg=msg,
        jobs=jobs,
    )


@app.route("/jobs/<job_id>", methods=["GET"])
def job_view(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        return ("Job not found", 404)

    return render_template_string(JOB_TEMPLATE, job=job)


# -----------------------------
# APIs
# -----------------------------
@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    with JOBS_LOCK:
        jobs = list(JOBS.values())

    # Keep list response lightweight
    out = []
    for j in jobs:
        out.append(
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "created_at": j["created_at"],
                "started_at": j.get("started_at"),
                "finished_at": j.get("finished_at"),
                "total_items": j.get("total_items"),
                "completed_items": j.get("completed_items"),
                "escalate_count": j.get("escalate_count"),
                "release_count": j.get("release_count"),
                "avg_score": j.get("avg_score"),
                "max_score": j.get("max_score"),
                "error": j.get("error"),
            }
        )

    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return jsonify({"kpis": _kpis_snapshot(), "jobs": out})


@app.route("/api/jobs/<job_id>", methods=["GET"])
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job)


@app.route("/api/jobs/<job_id>/audit", methods=["GET"])
def api_job_audit(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify({"job_id": job_id, "audit": job.get("audit") or []})


if __name__ == "__main__":
    # Avoid clashing with v3 default (5092)
    app.run(host="127.0.0.1", port=5094, debug=True)

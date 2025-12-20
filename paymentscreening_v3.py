# payment_screening_app.py
from flask import Flask, request, render_template_string, jsonify, url_for
import re, json
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Dummy sanctions / watchlist
# -----------------------------
WATCHLIST = [
    {
        "name": "Mohammad Al Hamed",
        "aka": ["Mohammed Al-Hameed", "Mohamad Alhammad"],
        "address": "12 King Faisal Road, Manama, Bahrain",
        "country": "BH",
        "dob": "1978-04-09",
        "list": "UN Sanctions",
        "category": "Terrorism"
    },
    {
        "name": "Zhang Wei",
        "aka": ["Wei Chang", "Z. Wei"],
        "address": "66 Nanjing West Road, Jing'an, Shanghai, China",
        "country": "CN",
        "dob": "1983-11-23",
        "list": "OFAC SDN",
        "category": "Proliferation"
    },
    {
        "name": "Hafiz Mohammed",
        "aka": ["Karachi", "Pakistan"],
        "address": "Karachi, Pakistan",
        "country": "PK",
        "dob": "1990-02-01",
        "list": "EU Consolidated",
        "category": "Corruption"
    },
    {
        "name": "Global Trade LLC",
        "aka": ["Global Trading Limited", "Global Trade Co."],
        "address": "PO Box 12345, Dubai, United Arab Emirates",
        "country": "AE",
        "dob": None,
        "list": "Internal Watch",
        "category": "Adverse Media"
    },
]

# -----------------------------
# Sanctioned countries (incl. alias)
# -----------------------------
SANCTIONED_COUNTRIES_RAW = ["pakistan", "iran", "syria", "ukraine", "cuba", "south korea"]
SANCTION_ALIASES = {"ukraise": "ukraine", "u k r a i s e": "ukraine"}

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\.\,;:\-\(\)\[\]\/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonical_country(s: str) -> str:
    s = _norm(s)
    if not s:
        return ""
    return SANCTION_ALIASES.get(s, s)

SANCTIONED_COUNTRIES = set(canonical_country(c) for c in SANCTIONED_COUNTRIES_RAW)

def address_has_sanctioned_country(addr: str):
    text = _norm(addr)
    for sc in SANCTIONED_COUNTRIES:
        if re.search(rf"\b{re.escape(sc)}\b", text):
            return True, sc
    return False, None

def is_sanctioned_country(country_str: str):
    can = canonical_country(country_str)
    return (can in SANCTIONED_COUNTRIES), can

# -----------------------------
# Abbrev & helpers
# -----------------------------
ABBR = {
    r"\bst\b": "street", r"\bstr\b": "street", r"\brd\b": "road",
    r"\bave\b": "avenue", r"\bav\b": "avenue", r"\bblvd\b": "boulevard",
    r"\bln\b": "lane", r"\bp\.?\s*o\.?\s*box\b": "po box"
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

def tokenize(s: str):
    return [t for t in re.split(r"\s+", normalize_text(s)) if t]

def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

# -----------------------------
# Jaro / Jaro-Winkler
# -----------------------------
def jaro_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1 or "", s2 or ""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 0.0
    max_dist = max(len1, len2)//2 - 1

    s1_matches = [False]*len1
    s2_matches = [False]*len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end   = min(i + max_dist + 1, len2)
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
    jaro = (matches/len1 + matches/len2 + (matches - transpositions)/matches) / 3.0
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
# Composite scoring
# -----------------------------
def name_similarity(input_name: str, wl_name: str, wl_aka: list) -> float:
    candidates = [wl_name] + (wl_aka or [])
    sims = []
    for cand in candidates:
        sims.append(jaro_winkler(input_name, cand))
        sims.append(0.5 * jaccard(tokenize(input_name), tokenize(cand)))
    return max(sims) if sims else 0.0

def address_similarity(input_addr: str, wl_addr: str) -> float:
    jw = jaro_winkler(input_addr, wl_addr)
    jac = jaccard(tokenize(input_addr), tokenize(wl_addr))
    return 0.4 * jw + 0.6 * jac

def dob_similarity(input_dob: str, wl_dob: str) -> float:
    if not input_dob or not wl_dob:
        return 0.0
    try:
        d1 = datetime.strptime(input_dob, "%Y-%m-%d").date()
        d2 = datetime.strptime(wl_dob, "%Y-%m-%d").date()
        return 1.0 if d1 == d2 else 0.0
    except Exception:
        return 0.0

def country_bonus(input_country: str, wl_country: str) -> float:
    if not input_country or not wl_country:
        return 0.0
    return 0.05 if _norm(input_country).upper() == _norm(wl_country).upper() else 0.0

def composite_risk_score(name_in, addr_in, country_in, dob_in, wl):
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
        "sanction_address_match": addr_match
    }

# -----------------------------
# Decisions
# -----------------------------
THRESHOLD = 0.80

def screen_payment(payload):
    candidates = []
    sanction_flag_any = False
    sanction_reasons = []

    for wl in WATCHLIST:
        s_payer, bd_payer = composite_risk_score(
            payload["payer_name"], payload["payer_address"], payload["payer_country"], payload["payer_dob"], wl
        )
        candidates.append(("PAYER", wl, s_payer, bd_payer))

        s_benef, bd_benef = composite_risk_score(
            payload["benef_name"], payload["benef_address"], payload["benef_country"], payload["benef_dob"], wl
        )
        candidates.append(("BENEFICIARY", wl, s_benef, bd_benef))

        for bd, role in [(bd_payer, "PAYER"), (bd_benef, "BENEFICIARY")]:
            if bd["sanction_party_country"]:
                sanction_flag_any = True
                sanction_reasons.append(f"{role} country in sanctioned list: {bd['sanction_party_country_name']}")
            if bd["sanction_address_hit"]:
                sanction_flag_any = True
                sanction_reasons.append(f"{role} address mentions sanctioned country: {bd['sanction_address_match']}")

    best = max(candidates, key=lambda x: x[2]) if candidates else None
    role, wl, score, breakdown = best if best else (None, None, 0.0, {})

    if sanction_flag_any:
        decision = "ESCALATE"; reason = "Sanctioned Country"
    else:
        decision = "ESCALATE" if score >= THRESHOLD else "RELEASE"
        reason = "Score Threshold" if decision == "ESCALATE" else "Below Threshold"

    sorted_cands = sorted(
        [{"role": r, "wl": w, "score": s, "breakdown": bd} for r, w, s, bd in candidates],
        key=lambda z: z["score"], reverse=True
    )

    return decision, reason, role, wl, score, breakdown, sanction_flag_any, sanction_reasons, sorted_cands

# -----------------------------
# On-prem explanation engine
# -----------------------------
def _extract_json_blocks(ctx: str):
    """Pull out JSON objects from the freeform context."""
    blocks = []
    # naive braces matching for top-level JSON objects
    stack = 0
    start = None
    for i, ch in enumerate(ctx):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0 and start is not None:
                snippet = ctx[start:i+1]
                try:
                    blocks.append(json.loads(snippet))
                except Exception:
                    pass
                start = None
    return blocks

def _best_result(json_blocks):
    """Try to find a dict that looks like your result object."""
    for b in json_blocks:
        if isinstance(b, dict) and (
            "decision" in b or ("best_score" in b and "candidates" in b)
        ):
            return b
    return None

def _score_key_drivers(breakdown: dict):
    parts = []
    if not isinstance(breakdown, dict):
        return parts
    for k in ("name", "address", "dob", "country"):
        if k in breakdown:
            try:
                parts.append((k, float(breakdown[k])))
            except Exception:
                pass
    parts.sort(key=lambda x: x[1], reverse=True)
    return parts

def local_explain(context: str) -> str:
    """
    Fully local ‘explanation’ generator (no network).
    It parses any JSON from the provided context, then crafts an audit-ready note.
    """
    blocks = _extract_json_blocks(context or "")
    res = _best_result(blocks) or {}

    decision = res.get("decision")
    reason = res.get("reason")
    best_score = res.get("best_score")
    best_role = res.get("best_role")
    best_wl = res.get("best_wl") or {}
    breakdown = res.get("breakdown") or {}
    sanction_flag = res.get("sanction_flag") or False
    sanction_reasons = res.get("sanction_reasons") or []

    # Human titles
    title = "Payment Screening Explanation (Local)"
    summary = []
    if decision:
        summary.append(f"Decision: {decision}")
    if reason:
        summary.append(f"Reason: {reason}")
    if best_score is not None:
        try:
            summary.append(f"Best match score: {float(best_score):.3f}")
        except Exception:
            summary.append(f"Best match score: {best_score}")

    # Key drivers
    drivers = _score_key_drivers(breakdown)
    driver_lines = [f"{name.capitalize()}: {val:.3f}" for name, val in drivers[:4]] or ["No driver details available."]

    # Watchlist context
    wl_bits = []
    if isinstance(best_wl, dict) and best_wl:
        wl_name = best_wl.get("name", "—")
        wl_list = best_wl.get("list", "—")
        wl_cat  = best_wl.get("category", "—")
        wl_ctry = best_wl.get("country", "—")
        wl_dob  = best_wl.get("dob", "—")
        wl_bits.append(f"Best match: {wl_name} (List: {wl_list}; Category: {wl_cat}; Country: {wl_ctry}; DOB: {wl_dob})")

    # Sanctions
    sanc_lines = []
    if sanction_flag:
        sanc_lines.append("Sanctions hit detected.")
        for r in sanction_reasons:
            sanc_lines.append(f"- {r}")
    else:
        sanc_lines.append("No sanctions hit detected from context.")

    # Actions
    if decision == "ESCALATE":
        actions = [
            "Place payment on hold and route to Level-2 review.",
            "Verify identity against authoritative KYC/KYB sources and documentary evidence.",
            "Re-screen name, address and country with up-to-date lists and adverse media.",
            "If sanctions flags are confirmed, follow blocking/reporting procedures."
        ]
    else:
        actions = [
            "Proceed with payment per standard STP rules.",
            "Retain screening logs, scores, and evidence for audit.",
            "Monitor for list updates; re-screen if new alerts occur."
        ]

    # Compose output
    out = []
    out.append(title)
    out.append("=" * len(title))
    if summary: out.append(" • " + " | ".join(summary))
    if wl_bits:
        out.append("\nWatchlist Context")
        out.append("-----------------")
        out.extend(wl_bits)
    out.append("\nKey Drivers")
    out.append("-----------")
    out.extend(f"- {line}" for line in driver_lines)
    out.append("\nSanctions")
    out.append("---------")
    out.extend(sanc_lines)
    out.append("\nRecommended Actions")
    out.append("-------------------")
    out.extend(f"- {a}" for a in actions)
    return "\n".join(out)

# -----------------------------
# Floating Explain Endpoint (On-prem)
# -----------------------------
@app.route("/ai/explain", methods=["POST", "OPTIONS"])
def ai_explain():
    # Preflight for responsiveness
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": request.headers.get("Origin", "*"),
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        })

    data = request.get_json(force=True, silent=True) or {}
    # model is ignored now (kept for UI compatibility)
    context = (data.get("context") or "").strip()
    try:
        text = local_explain(context)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"Local explain error: {e}"}), 500

# -----------------------------
# Template w/ Floating Panel (unchanged UI)
# -----------------------------
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bank Payment Screening & Compliance (Demo)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #0f172a; color: #e2e8f0; }
    .card { border-radius: 1rem; box-shadow: 0 10px 25px rgba(0,0,0,.3); }
    .card-header { background: linear-gradient(90deg, #1e293b, #0f172a); color: #cbd5e1; }
    .badge-slim { font-size: .75rem; }
    .score-pill { font-weight: 700; }
    .decision-release { background: #0ea5e9; }
    .decision-escalate { background: #f59e0b; }
    .table-dark tbody tr td { color: #e2e8f0; }
    .muted { color: #94a3b8; }
    .mini { font-size: .9rem; }
    a,a:visited { color: #93c5fd; }

    /* Floating Explain panel */
    #aiToggle {
      position: fixed; right: 16px; bottom: 16px; z-index: 1030;
      border-radius: 999px;
    }
    #aiPanel {
      position: fixed; right: 16px; bottom: 80px; z-index: 1030;
      width: min(460px, 92vw);
      display: none;
      background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 12px;
      box-shadow: 0 20px 40px rgba(0,0,0,.45);
    }
    #aiPanel .hdr {
      background: #1f2937; border-bottom: 1px solid #374151; border-top-left-radius: 12px; border-top-right-radius: 12px;
      padding: .75rem 1rem;
    }
    #aiPanel .body { padding: .75rem 1rem; }
    #aiOut {
      display: none; background: #0b1220; color: #dbeafe; border: 1px solid #1f2937;
      border-radius: 8px; padding: .75rem; white-space: pre-wrap; max-height: 260px; overflow: auto;
    }
  </style>
</head>
<body>
<div class="container py-4">
  <div class="card border-0">
    <div class="card-header">
      <h3 class="m-0">Payment Screening & Compliance (Sample)</h3>
      <div class="muted mini">Dummy data • Jaro-Winkler + Jaccard matching • Threshold: 0.80 → Escalate</div>
      <div class="muted mini">
        Sanction list (as provided):
        {% for c in sanctioned_display %}<span class="badge text-bg-dark me-1">{{ c }}</span>{% endfor %}
      </div>
      <div class="muted mini">Hard rule: If party country is sanctioned or address mentions a sanctioned country → <b>ESCALATE (Sanctioned Country)</b></div>
    </div>

    <div class="card-body bg-dark">
      <form method="post" class="row g-3">
        <div class="col-12"><h5 class="text-info">Payer</h5></div>
        <div class="col-md-4">
          <label class="form-label">Name</label>
          <input class="form-control" name="payer_name" value="{{form.payer_name or 'Global Trade LLC'}}" required>
        </div>
        <div class="col-md-5">
          <label class="form-label">Address</label>
          <input class="form-control" name="payer_address" value="{{form.payer_address or 'PO Box 12345, Dubai, UAE'}}" required>
        </div>
        <div class="col-md-1">
          <label class="form-label">Country</label>
          <input class="form-control" name="payer_country" value="{{form.payer_country or 'AE'}}" required>
        </div>
        <div class="col-md-2">
          <label class="form-label">DOB (YYYY-MM-DD)</label>
          <input class="form-control" name="payer_dob" value="{{form.payer_dob or ''}}">
        </div>

        <div class="col-12 pt-2"><h5 class="text-info">Beneficiary</h5></div>
        <div class="col-md-4">
          <label class="form-label">Name</label>
          <input class="form-control" name="benef_name" value="{{form.benef_name or 'Olena Petrenko-Kovalenko'}}" required>
        </div>
        <div class="col-md-5">
          <label class="form-label">Address</label>
          <input class="form-control" name="benef_address" value="{{form.benef_address or 'Kreschatyk 22, Kyiv, Ukraine'}}" required>
        </div>
        <div class="col-md-1">
          <label class="form-label">Country</label>
          <input class="form-control" name="benef_country" value="{{form.benef_country or 'UA'}}" required>
        </div>
        <div class="col-md-2">
          <label class="form-label">DOB (YYYY-MM-DD)</label>
          <input class="form-control" name="benef_dob" value="{{form.benef_dob or '1990-02-01'}}">
        </div>

        <div class="col-md-2">
          <label class="form-label">Amount</label>
          <input class="form-control" name="amount" type="number" step="0.01" value="{{form.amount or 12500.00}}" required>
        </div>
        <div class="col-md-1">
          <label class="form-label">CCY</label>
          <input class="form-control" name="currency" value="{{form.currency or 'USD'}}" required>
        </div>
        <div class="col-md-9">
          <label class="form-label">Reference</label>
          <input class="form-control" name="reference" value="{{form.reference or 'Invoice 2025-10-ACME'}}">
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-primary px-4">Screen Payment</button>
          <a class="btn btn-outline-light ms-2" href="{{url_for('home')}}">Reset</a>
        </div>
      </form>

      {% if result %}
      <hr class="border-secondary">
      <div class="row g-3 align-items-center">
        <div class="col-md-4">
          <div class="p-3 rounded-3 {{ 'decision-escalate' if result.decision=='ESCALATE' else 'decision-release' }}">
            <div class="text-dark fw-bold">Decision</div>
            <div class="fs-3 fw-bolder">{{ result.decision }}</div>
            <div class="mini text-dark">
              Reason: {{ result.reason }} • Best score:
              <span class="score-pill">{{ '%.3f'|format(result.best_score) }}</span>
            </div>
            {% if result.sanction_flag %}
              <div class="mini text-dark mt-2"><b>Sanction hits:</b>
                <ul class="m-0">
                  {% for r in result.sanction_reasons %}<li>{{ r }}</li>{% endfor %}
                </ul>
              </div>
            {% endif %}
          </div>
        </div>
        <div class="col-md-8">
          <div class="p-3 bg-secondary rounded-3">
            <div class="fw-bold">Best Match ({{ result.best_role }}) → {{ result.best_wl.name }}</div>
            <div class="mini">List: {{ result.best_wl.list }} • Category: {{ result.best_wl.category }} • Country: {{ result.best_wl.country }} • DOB: {{ result.best_wl.dob or '—' }}</div>
            <div class="mini">Address: {{ result.best_wl.address }}</div>
            <div class="pt-2">
              <span class="badge bg-dark badge-slim">Name: {{ '%.3f'|format(result.breakdown.name) }}</span>
              <span class="badge bg-dark badge-slim">Address: {{ '%.3f'|format(result.breakdown.address) }}</span>
              <span class="badge bg-dark badge-slim">DOB: {{ '%.3f'|format(result.breakdown.dob) }}</span>
              <span class="badge bg-dark badge-slim">Country bonus: {{ '%.3f'|format(result.breakdown.country) }}</span>
              {% if result.breakdown.sanction_party_country %}
                <span class="badge bg-warning text-dark badge-slim">Party Country Sanction: {{ result.breakdown.sanction_party_country_name }}</span>
              {% endif %}
              {% if result.breakdown.sanction_address_hit %}
                <span class="badge bg-warning text-dark badge-slim">Address Sanction: {{ result.breakdown.sanction_address_match }}</span>
              {% endif %}
            </div>
          </div>
        </div>
      </div>

      <div class="pt-4">
        <h5>All Candidate Matches</h5>
        <div class="table-responsive">
          <table class="table table-dark table-striped align-middle">
            <thead><tr>
              <th>#</th><th>Role</th><th>Watchlist Name</th><th>List</th><th>Country</th><th>Score</th>
              <th>Name</th><th>Address</th><th>DOB</th><th>Country Bonus</th><th>Sanction?</th><th>Sanction Detail</th>
            </tr></thead>
            <tbody>
              {% for row in result.candidates %}
              {% set idx = loop.index %}
              {% set sanc = (row.breakdown.sanction_party_country or row.breakdown.sanction_address_hit) %}
              <tr class="{{ 'table-warning text-dark' if sanc else '' }}">
                <td>{{ idx }}</td>
                <td>{{ row.role }}</td>
                <td>{{ row.wl.name }}</td>
                <td>{{ row.wl.list }}</td>
                <td>{{ row.wl.country }}</td>
                <td><b>{{ '%.3f'|format(row.score) }}</b></td>
                <td>{{ '%.3f'|format(row.breakdown.name) }}</td>
                <td>{{ '%.3f'|format(row.breakdown.address) }}</td>
                <td>{{ '%.3f'|format(row.breakdown.dob) }}</td>
                <td>{{ '%.3f'|format(row.breakdown.country) }}</td>
                <td>{{ 'YES' if sanc else 'No' }}</td>
                <td>
                  {% if row.breakdown.sanction_party_country %}Party: {{ row.breakdown.sanction_party_country_name }}{% endif %}
                  {% if row.breakdown.sanction_address_hit %}{% if row.breakdown.sanction_party_country %} • {% endif %}Address: {{ row.breakdown.sanction_address_match }}{% endif %}
                  {% if not sanc %}—{% endif %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
        <div class="muted mini">
          Heuristics: Score = 0.60·Name + 0.30·Address + 0.05·DOB + CountryBonus (0.05 if exact).
          Name = max(Jaro-Winkler, 0.5·Jaccard on tokens) across primary & AKAs. Address = 0.4·Jaro-Winkler + 0.6·Jaccard.
          <br><b>Hard rule:</b> If party country is sanctioned or an address mentions a sanctioned country → <b>ESCALATE (Sanctioned Country)</b>.
        </div>
      </div>
      {% endif %}
    </div>
  </div>

  <!-- Floating Explain Button -->
  <button id="aiToggle" class="btn btn-info">AI Explain</button>

  <!-- Floating Explain Panel (kept intact) -->
  <div id="aiPanel">
    <div class="hdr d-flex justify-content-between align-items-center">
      <div><b>AI Explain (On-prem)</b></div>
      <button id="aiClose" class="btn btn-sm btn-outline-light">Close</button>
    </div>
    <div class="body">
      <div class="mb-2">
        <label class="form-label">Model (ignored locally)</label>
        <select id="aiModel" class="form-select form-select-sm">
          <option value="local">local-explainer</option>
          <option value="gpt-4o-mini">gpt-4o-mini (disabled)</option>
        </select>
      </div>
      <div class="mb-2">
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="checkbox" id="ctxForm" checked>
          <label class="form-check-label" for="ctxForm">Include form</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="checkbox" id="ctxResult" checked>
          <label class="form-check-label" for="ctxResult">Include screening result</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="checkbox" id="ctxSanctions">
          <label class="form-check-label" for="ctxSanctions">Include sanctioned countries</label>
        </div>
      </div>
      <textarea id="ctxFree" class="form-control form-control-sm" rows="2" placeholder="Optional notes for the explainer..."></textarea>
      <div class="mt-2 d-flex align-items-center gap-2">
        <button id="aiGenerate" class="btn btn-primary btn-sm">Generate</button>
        <button id="aiAddContext" class="btn btn-success btn-sm">Add Latest Context</button>
        <span id="aiStatus" class="mini"></span>
      </div>
      <pre id="aiOut" class="mt-2"></pre>
    </div>
  </div>
</div>

<script>
  // Floating panel toggles
  const aiToggle = document.getElementById('aiToggle');
  const aiPanel  = document.getElementById('aiPanel');
  const aiClose  = document.getElementById('aiClose');
  aiToggle.onclick = () => { aiPanel.style.display = (aiPanel.style.display === 'block' ? 'none' : 'block'); };
  aiClose.onclick  = () => { aiPanel.style.display = 'none'; };

  const SERVER_RESULT    = {{ (result or {})|tojson }};
  const SERVER_SANCTIONS = {{ sanctioned_display|tojson }};

  function collectFormContext() {
    const data = {};
    document.querySelectorAll('form [name]').forEach(el => { data[el.name] = el.value; });
    return data;
    }
  function buildContext() {
    const parts = [];
    if (document.getElementById('ctxForm').checked) {
      parts.push("Form:\\n" + JSON.stringify(collectFormContext(), null, 2));
    }
    if (document.getElementById('ctxResult').checked && SERVER_RESULT && Object.keys(SERVER_RESULT).length) {
      parts.push("Screening Result:\\n" + JSON.stringify(SERVER_RESULT, null, 2));
    }
    if (document.getElementById('ctxSanctions').checked) {
      parts.push("Sanctioned Countries:\\n" + JSON.stringify(SERVER_SANCTIONS, null, 2));
    }
    const extra = document.getElementById('ctxFree').value.trim();
    if (extra) parts.push("Notes:\\n" + extra);
    return parts.join("\\n\\n");
  }

  const genBtn = document.getElementById('aiGenerate');
  const addContextBtn = document.getElementById('aiAddContext');
  const status = document.getElementById('aiStatus');
  const out    = document.getElementById('aiOut');

  addContextBtn.onclick = () => {
    const ctx = buildContext();
    const currentNotes = document.getElementById('ctxFree').value.trim();
    const timestamp = new Date().toLocaleString();
    const newNotes = currentNotes 
      ? `${currentNotes}\n\n--- Latest Context (${timestamp}) ---\n${ctx}`
      : `--- Latest Context (${timestamp}) ---\n${ctx}`;
    document.getElementById('ctxFree').value = newNotes;
    status.textContent = "Context added to notes";
    setTimeout(() => { status.textContent = ""; }, 2000);
  };

  genBtn.onclick = async () => {
    const model = document.getElementById('aiModel').value; // ignored locally
    const ctx = buildContext();
    genBtn.disabled = true;
    genBtn.textContent = "Generating…";
    status.textContent = "Running on-prem explainer…";
    out.style.display = 'none';
    out.textContent = "";

    try {
      const res = await fetch("{{ url_for('ai_explain') }}", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ model, context: ctx })
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "Unknown error");
      out.textContent = j.text || "(empty)";
      out.style.display = 'block';
      status.textContent = "Done.";
    } catch (err) {
      status.textContent = "Error: " + err.message;
    } finally {
      genBtn.disabled = false;
      genBtn.textContent = "Generate";
    }
  };
</script>
</body>
</html>
"""

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    form = {
        "payer_name": request.form.get("payer_name", ""),
        "payer_address": request.form.get("payer_address", ""),
        "payer_country": request.form.get("payer_country", ""),
        "payer_dob": request.form.get("payer_dob", ""),

        "benef_name": request.form.get("benef_name", ""),
        "benef_address": request.form.get("benef_address", ""),
        "benef_country": request.form.get("benef_country", ""),
        "benef_dob": request.form.get("benef_dob", ""),

        "amount": request.form.get("amount", ""),
        "currency": request.form.get("currency", ""),
        "reference": request.form.get("reference", ""),
    }

    result = None
    if request.method == "POST":
        payload = {
            "payer_name": form["payer_name"],
            "payer_address": form["payer_address"],
            "payer_country": form["payer_country"],
            "payer_dob": form["payer_dob"],
            "benef_name": form["benef_name"],
            "benef_address": form["benef_address"],
            "benef_country": form["benef_country"],
            "benef_dob": form["benef_dob"],
            "amount": float(form["amount"]) if form["amount"] else 0.0,
            "currency": form["currency"],
            "reference": form["reference"],
        }
        decision, reason, role, wl, score, breakdown, sanc_flag, sanc_reasons, candidates = screen_payment(payload)

        class Obj(dict):
            __getattr__ = dict.get
        result = Obj({
            "decision": decision,
            "reason": reason,
            "best_role": role,
            "best_wl": wl,
            "best_score": score,
            "breakdown": Obj(breakdown),
            "sanction_flag": sanc_flag,
            "sanction_reasons": sanc_reasons,
            "candidates": [Obj({
                "role": c["role"],
                "wl": c["wl"],
                "score": c["score"],
                "breakdown": Obj(c["breakdown"])
            }) for c in candidates]
        })

    return render_template_string(
        TEMPLATE,
        form=form,
        result=result,
        sanctioned_display=SANCTIONED_COUNTRIES_RAW
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5092, debug=True)

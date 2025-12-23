# payment_screening_api.py
from flask import Flask, request, jsonify
import re
import json
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
# Screening Logic
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
        decision = "ESCALATE"
        reason = "Sanctioned Country"
    else:
        decision = "ESCALATE" if score >= THRESHOLD else "RELEASE"
        reason = "Score Threshold" if decision == "ESCALATE" else "Below Threshold"

    sorted_cands = sorted(
        [{"role": r, "wl": w, "score": s, "breakdown": bd} for r, w, s, bd in candidates],
        key=lambda z: z["score"], reverse=True
    )

    return {
        "decision": decision,
        "reason": reason,
        "best_role": role,
        "best_wl": wl,
        "best_score": score,
        "breakdown": breakdown,
        "sanction_flag": sanction_flag_any,
        "sanction_reasons": sanction_reasons,
        "candidates": sorted_cands
    }

# -----------------------------
# API Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Payment Screening API",
        "version": "1.0",
        "endpoints": {
            "/screen": {
                "method": "POST",
                "description": "Screen a payment transaction",
                "required_fields": [
                    "payer_name", "payer_address", "payer_country",
                    "benef_name", "benef_address", "benef_country",
                    "amount", "currency"
                ]
            },
            "/watchlist": {
                "method": "GET",
                "description": "Get the current watchlist"
            },
            "/sanctioned-countries": {
                "method": "GET",
                "description": "Get the list of sanctioned countries"
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/screen", methods=["POST"])
def screen():
    """
    Screen a payment transaction
    Expects JSON payload with payer and beneficiary details
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            "payer_name", "payer_address", "payer_country",
            "benef_name", "benef_address", "benef_country",
            "amount", "currency"
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400
        
        # Prepare payload with optional fields
        payload = {
            "payer_name": data["payer_name"],
            "payer_address": data["payer_address"],
            "payer_country": data["payer_country"],
            "payer_dob": data.get("payer_dob", ""),
            "benef_name": data["benef_name"],
            "benef_address": data["benef_address"],
            "benef_country": data["benef_country"],
            "benef_dob": data.get("benef_dob", ""),
            "amount": float(data["amount"]),
            "currency": data["currency"],
            "reference": data.get("reference", "")
        }
        
        # Screen the payment
        result = screen_payment(payload)
        
        # Add request metadata to response
        response = {
            "timestamp": datetime.now().isoformat(),
            "screening_result": result,
            "transaction_details": {
                "amount": payload["amount"],
                "currency": payload["currency"],
                "reference": payload["reference"]
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """
    Get the current watchlist entries
    """
    return jsonify({
        "watchlist": WATCHLIST,
        "total_entries": len(WATCHLIST)
    })

@app.route("/sanctioned-countries", methods=["GET"])
def get_sanctioned_countries():
    """
    Get the list of sanctioned countries
    """
    return jsonify({
        "sanctioned_countries": SANCTIONED_COUNTRIES_RAW,
        "aliases": SANCTION_ALIASES,
        "total_countries": len(SANCTIONED_COUNTRIES)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The HTTP method is not supported for this endpoint"
    }), 405

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Starting Payment Screening API...")
    print("API Documentation available at: http://127.0.0.1:5000/")
    print("Screen payments at: POST http://127.0.0.1:5000/screen")
    app.run(host="127.0.0.1", port=5000, debug=True)

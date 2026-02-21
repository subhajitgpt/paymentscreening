from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# -----------------------------
# Demo watchlist / sanctions
# -----------------------------

WATCHLIST: List[Dict[str, Any]] = [
    {
        "name": "Mohammad Al Hamed",
        "aka": ["Mohammed Al-Hameed", "Mohamad Alhammad"],
        "address": "12 King Faisal Road, Manama, Bahrain",
        "country": "BH",
        "dob": "1978-04-09",
        "list": "UN Sanctions",
        "category": "Terrorism",
    },
    {
        "name": "Zhang Wei",
        "aka": ["Wei Chang", "Z. Wei"],
        "address": "66 Nanjing West Road, Jing'an, Shanghai, China",
        "country": "CN",
        "dob": "1983-11-23",
        "list": "OFAC SDN",
        "category": "Proliferation",
    },
    {
        "name": "Hafiz Mohammed",
        "aka": ["Karachi", "Pakistan"],
        "address": "Karachi, Pakistan",
        "country": "PK",
        "dob": "1990-02-01",
        "list": "EU Consolidated",
        "category": "Corruption",
    },
    {
        "name": "Global Trade LLC",
        "aka": ["Global Trading Limited", "Global Trade Co."],
        "address": "PO Box 12345, Dubai, United Arab Emirates",
        "country": "AE",
        "dob": None,
        "list": "Internal Watch",
        "category": "Adverse Media",
    },
]

# Country names (not codes) are easiest to match in free-text addresses.
SANCTIONED_COUNTRIES_RAW = ["pakistan", "iran", "syria", "ukraine", "cuba", "south korea"]
SANCTION_ALIASES = {"ukraise": "ukraine", "u k r a i s e": "ukraine"}


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\.,;:\-\(\)\[\]/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _try_country_code_to_name(country_str: str) -> Optional[str]:
    # Optional dependency: pycountry (already in requirements.txt)
    try:
        import pycountry  # type: ignore

        code = (country_str or "").strip().upper()
        if len(code) == 2 and code.isalpha():
            c = pycountry.countries.get(alpha_2=code)
            if c:
                # Prefer common_name when available; fall back to name.
                name = getattr(c, "common_name", None) or getattr(c, "name", None)
                if isinstance(name, str) and name.strip():
                    return name
    except Exception:
        return None
    return None


def canonical_country(country_str: str) -> str:
    """Canonicalize an input country.

    - Accepts either free-text ("Ukraine") or ISO alpha-2 codes ("UA").
    - Normalizes punctuation/spacing.
    - Applies known aliases.

    Returns a lowercase canonical name-like token suitable for matching against
    SANCTIONED_COUNTRIES.
    """

    if not country_str:
        return ""

    code_name = _try_country_code_to_name(country_str)
    s = _norm(code_name or country_str)
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
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


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
    return 0.05 if canonical_country(input_country) == canonical_country(wl_country) else 0.0


def composite_risk_score(
    name_in: str,
    addr_in: str,
    country_in: str,
    dob_in: str,
    wl: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    n = name_similarity(name_in, str(wl.get("name") or ""), wl.get("aka"))
    a = address_similarity(addr_in, str(wl.get("address") or ""))
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


REQUIRED_FIELDS = [
    "payer_name",
    "payer_address",
    "payer_country",
    "payer_dob",
    "benef_name",
    "benef_address",
    "benef_country",
    "benef_dob",
]


def validate_payload(payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    missing = [k for k in REQUIRED_FIELDS if k not in payload]
    return (len(missing) == 0), missing


def coerce_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
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


@dataclass(frozen=True)
class Candidate:
    role: str
    watchlist: Dict[str, Any]
    score: float
    breakdown: Dict[str, Any]


def build_candidates(payload: Mapping[str, Any]) -> Tuple[List[Candidate], bool, List[str]]:
    candidates: List[Candidate] = []
    sanction_flag_any = False
    sanction_reasons: List[str] = []

    for wl in WATCHLIST:
        s_payer, bd_payer = composite_risk_score(
            str(payload.get("payer_name") or ""),
            str(payload.get("payer_address") or ""),
            str(payload.get("payer_country") or ""),
            str(payload.get("payer_dob") or ""),
            wl,
        )
        candidates.append(Candidate(role="PAYER", watchlist=dict(wl), score=float(s_payer), breakdown=bd_payer))

        s_benef, bd_benef = composite_risk_score(
            str(payload.get("benef_name") or ""),
            str(payload.get("benef_address") or ""),
            str(payload.get("benef_country") or ""),
            str(payload.get("benef_dob") or ""),
            wl,
        )
        candidates.append(
            Candidate(role="BENEFICIARY", watchlist=dict(wl), score=float(s_benef), breakdown=bd_benef)
        )

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

    return candidates, sanction_flag_any, sanction_reasons


def pick_best_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def apply_decision_policy(best_score: float, sanction_flag_any: bool) -> Tuple[str, str]:
    if sanction_flag_any:
        return "ESCALATE", "Sanctioned Country"

    if float(best_score or 0.0) >= THRESHOLD:
        return "ESCALATE", "Score Threshold"

    return "RELEASE", "Below Threshold"

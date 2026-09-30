"""Verification: is every figure in an answer traceable to data the agent retrieved?

Numbers are pulled from the answer and matched against every number in the tool
results (and the user's question). A figure matches if it equals a source value
within rounding, allowing for sign, percent-versus-fraction and unit suffixes.
"""
import json, re

NUM = re.compile(r"(?<![\w.])([-−‑–])?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|k\b|K\b|M\b|million\b|billion\b|ha\b|acres?\b)?")
CITATION = re.compile(r"\[D\d+\]|\bp\.\s*\d+|\bpage \d+", re.I)


def _figures(text: str) -> list[dict]:
    text = CITATION.sub(" ", text)
    out = []
    for m in NUM.finditer(text):
        sign, whole, frac, unit = m.groups()
        value = float(whole.replace(",", "") + (frac or ""))
        if sign:
            value = -value
        unit = (unit or "").lower()
        if unit in ("k",):
            value *= 1e3
        elif unit in ("m", "million"):
            value *= 1e6
        elif unit == "billion":
            value *= 1e9
        decimals = len(frac) - 1 if frac else 0
        out.append({"text": m.group(0).strip(), "value": value, "decimals": decimals, "unit": unit, "signed": bool(sign)})
    return out


def _skip(f: dict) -> bool:
    v = f["value"]
    if f["decimals"] == 0 and not f["unit"]:
        if 1900 <= v <= 2100:      # years
            return True
        if abs(v) <= 10:           # counts and list numbers ("3 counties", "step 2")
            return True
    return False


def _source_numbers(obj, acc: set):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        for f in _figures(obj):
            acc.add(f["value"])
    elif isinstance(obj, dict):
        for v in obj.values():
            _source_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _source_numbers(v, acc)


def _matches(f: dict, sources: set) -> float | None:
    v = f["value"]
    tol_round = 0.5 * 10 ** (-f["decimals"])
    for s in sources:
        for cand in (s, s * 100, s / 100):
            tol = max(tol_round, 0.005 * abs(cand))
            # A signed figure ("-44%") must match a source of the same sign; an unsigned one
            # ("fell 44%") may match either sign.
            diff = abs(v - cand) if f.get("signed") else abs(abs(v) - abs(cand))
            if diff <= tol:
                return s
    return None


def verify_answer(answer: str, tool_results: list, question: str = "") -> dict:
    sources: set = set()
    _source_numbers(tool_results, sources)
    for f in _figures(question):
        sources.add(f["value"])
    checked, verified, unverified = [], [], []
    for f in _figures(answer):
        if _skip(f):
            continue
        match = _matches(f, sources)
        row = {"figure": f["text"], "matched": match is not None, "source_value": match}
        checked.append(row)
        (verified if match is not None else unverified).append(f["text"])
    n = len(checked)
    return {
        "figures_checked": n,
        "figures_verified": len(verified),
        "share_verified": round(len(verified) / n, 3) if n else 1.0,
        "unverified": unverified,
        "details": checked,
    }

import datetime
import ee
import json
import os
import time
import requests as http
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from pydantic import BaseModel
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── GEE init — three paths: service account → oauth token → app-default ───────
# Server starts even if all GEE auth fails; fire/India/El Niño work without GEE.
_gee_available = False
try:
    _gee_project = os.getenv("GEE_PROJECT", "kelpwatch-2026")
    _gee_sa_key  = os.getenv("GEE_SERVICE_ACCOUNT_KEY", "")
    _gee_creds   = os.getenv("GEE_CREDENTIALS_JSON", "")

    # Also support a file path (local dev — avoids dotenv multiline issues)
    _gee_key_file = os.getenv("GEE_KEY_FILE", "")
    if not _gee_sa_key and _gee_key_file:
        if not os.path.exists(_gee_key_file):
            raise FileNotFoundError(f"GEE_KEY_FILE={_gee_key_file} does not exist — add it as a Secret File")
        with open(_gee_key_file) as _f:
            _gee_sa_key = _f.read()

    if _gee_sa_key:
        _key_data = json.loads(_gee_sa_key)
        _creds = ee.ServiceAccountCredentials(
            email=_key_data["client_email"], key_data=_gee_sa_key
        )
        ee.Initialize(credentials=_creds, project=_gee_project)
        print("✅ GEE initialized via service account")
    elif _gee_creds:
        # Alternative: OAuth refresh token from `earthengine authenticate`
        import google.oauth2.credentials
        _cd = json.loads(_gee_creds)
        _oauth = google.oauth2.credentials.Credentials(
            token=None,
            refresh_token=_cd["refresh_token"],
            client_id=_cd["client_id"],
            client_secret=_cd["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(credentials=_oauth, project=_gee_project)
        print("✅ GEE initialized via OAuth refresh token")
    else:
        # Local dev: application-default credentials
        ee.Initialize(project=_gee_project)
        print("✅ GEE initialized via application-default credentials")

    _gee_available = True
except Exception as _gee_err:
    print(f"⚠ GEE unavailable — kelp tiles degraded. Fix: set GEE_KEY_FILE (secret file) or GEE_SERVICE_ACCOUNT_KEY.\n  {_gee_err}")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "groq")

# ── Supabase / auth config ────────────────────────────────────────────────────
SUPABASE_URL         = os.getenv("SUPABASE_URL", "").rstrip("/")
# SUPABASE_PRIVATE = new-style sb_secret_… key; SUPABASE_SERVICE_KEY = legacy service_role JWT
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_PRIVATE", "")
SUPABASE_JWT_SECRET  = os.getenv("SUPABASE_JWT_SECRET", "")  # only for legacy HS256 projects
DAILY_QUERY_LIMIT    = int(os.getenv("DAILY_QUERY_LIMIT", "5"))

def _sb_headers(json_body: bool = False) -> dict:
    """PostgREST headers. New sb_secret_ keys are not JWTs, so they go in `apikey` only."""
    hdrs = {"apikey": SUPABASE_SERVICE_KEY}
    if not SUPABASE_SERVICE_KEY.startswith("sb_"):
        hdrs["Authorization"] = f"Bearer {SUPABASE_SERVICE_KEY}"
    if json_body:
        hdrs["Content-Type"] = "application/json"
    return hdrs

_jwks: dict | None = None

def _supabase_jwks(refresh: bool = False) -> dict:
    """Public signing keys for projects on asymmetric JWT signing keys (cached)."""
    global _jwks
    if _jwks is None or refresh:
        r = http.get(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", timeout=5)
        r.raise_for_status()
        _jwks = r.json()
    return _jwks

def verify_token(authorization: str = Header(default="")) -> dict:
    """Verify Supabase JWT. Returns user payload. Bypasses only when Supabase is not configured (local dev)."""
    if not SUPABASE_URL:
        return {"sub": "local-dev"}
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign up or sign in to use the AI agent.")
    token = authorization.split(" ", 1)[1]
    try:
        alg = jwt.get_unverified_header(token).get("alg", "")
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise JWTError("HS256 token but SUPABASE_JWT_SECRET is not set")
            return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
        try:
            return jwt.decode(token, _supabase_jwks(), algorithms=["ES256", "RS256"], audience="authenticated")
        except JWTError:
            # Signing key may have rotated since we cached the JWKS
            return jwt.decode(token, _supabase_jwks(refresh=True), algorithms=["ES256", "RS256"], audience="authenticated")
    except (JWTError, http.RequestException):
        raise HTTPException(status_code=401, detail="Session expired — please sign in again.")

def check_and_increment_usage(user_id: str) -> int:
    """Enforce daily limit. Returns queries_remaining. Raises 429 when over limit."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or user_id == "local-dev":
        return DAILY_QUERY_LIMIT
    today = datetime.date.today().isoformat()
    hdrs = _sb_headers(json_body=True)
    r = http.get(
        f"{SUPABASE_URL}/rest/v1/token_usage",
        headers=hdrs,
        params={"user_id": f"eq.{user_id}", "query_date": f"eq.{today}", "select": "query_count"},
    )
    data = r.json() if r.ok else []
    current = data[0]["query_count"] if data else 0
    if current >= DAILY_QUERY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"You've used all {DAILY_QUERY_LIMIT} queries for today. Resets at midnight UTC."
        )
    if data:
        http.patch(
            f"{SUPABASE_URL}/rest/v1/token_usage", headers=hdrs,
            params={"user_id": f"eq.{user_id}", "query_date": f"eq.{today}"},
            json={"query_count": current + 1},
        )
    else:
        http.post(
            f"{SUPABASE_URL}/rest/v1/token_usage", headers=hdrs,
            json={"user_id": user_id, "query_date": today, "query_count": 1},
        )
    return DAILY_QUERY_LIMIT - current - 1

# ── App, data and agent wiring ────────────────────────────────────────────────
import pathlib
import climate
from agent import Agent, TOOL_SPECS
from kb import KnowledgeBase, run_eval
from policy import load_policy

KB = KnowledgeBase()
RETRIEVAL_EVAL = run_eval(KB)
climate.record("documents", True, [
    {"check": "documents_ingested", "passed": True, "detail": f"{KB.report['totals']['documents']} documents, {KB.report['totals']['chunks']} passages"},
    {"check": "no_unreadable_pages", "passed": KB.report["totals"]["unreadable_pages"] == 0,
     "detail": f"{KB.report['totals']['unreadable_pages']} image-only pages need OCR"},
    {"check": "chunk_sizes", "passed": KB.report["chunk_size_check"]["passed"],
     "detail": f"{KB.report['chunk_size_check']['within_range']} of {KB.report['chunk_size_check']['total']} passages within size limits"},
    {"check": "retrieval_eval", "passed": RETRIEVAL_EVAL["with_thesaurus"]["hit_rate"] >= 0.9,
     "detail": f"{RETRIEVAL_EVAL['with_thesaurus']['hits']} of {RETRIEVAL_EVAL['with_thesaurus']['total']} test questions retrieved correctly in the top {RETRIEVAL_EVAL['k']}"},
])

def _llm():
    """Any OpenAI-compatible chat API with tool calling. Choose with LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        return openai_client, os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return groq_client, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

def _compact_kelp(k: dict) -> dict:
    """What the agent needs from the kelp data, without per-cell check details (context budget)."""
    if k.get("error"):
        return k
    return {"metric": k["metric"], "computed_at": k["computed_at"], "status": k["status"], "method": k["method"],
            "counties": [{"county": c["county"],
                          "signal_pct_by_period": {t["period"]: t["signal_pct"] for t in c["timeline"]},
                          "change_2015_2023_pct": c["change_2015_2023_pct"],
                          "change_1995_2023_pct_not_comparable": c["change_1995_2023_pct"]} for c in k["counties"]],
            "checks": [{"check": c["check"], "passed": c["passed"], "detail": c["detail"]} for c in k["checks"]],
            "caveats": k["caveats"], "documented_trend": k["documented_trend"]["statement"],
            "quality_issues": k["quality_issues"]}

def _compact_yearly(k: dict) -> dict:
    if k.get("error"):
        return k
    return {"method": k["method"], "computed_at": k["computed_at"],
            "columns": ["year", "sensor", "signal_pct", "ndvi_mean"],
            "counties": {c: [[r["year"], r["sensor"], r["signal_pct"], r["ndvi_mean"]] for r in rows] for c, rows in k["counties"].items()},
            "checks": k["checks"], "quality_issues": k["quality_issues"]}

def _compact_sst(k: dict) -> dict:
    s = k.get("sst") if not k.get("error") else None
    if not s:
        return {"error": "Sea-surface temperature series unavailable", "quality_issues": ["sea temperature unavailable"]}
    return {"region": s["region"], "normal_1991_2020_c": s["normal_1991_2020_c"], "method": s["method"],
            "columns": ["year", "summer_sst_c", "anomaly_c"],
            "years": [[y["year"], y["summer_sst_c"], y["anomaly_c"]] for y in s["years"]],
            "caveats": s["caveats"], "quality_issues": []}

def _compact_enso(e: dict) -> dict:
    if e.get("error"):
        return e
    return {k: e[k] for k in ("computed_at", "oni_latest", "nino34_oisst_30d", "method", "thresholds", "caveats",
                              "pacific_northwest_note", "quality_issues", "status")} | {
            "oni_recent": [[r["season"], r["year"], r["oni"]] for r in e["oni_series"][-24:]]}

AGENT_TOOLS = {
    "get_kelp_trends": lambda county=None: _compact_kelp(climate.kelp(county)),
    "get_wildfire_activity": lambda: {k: v for k, v in climate.fire(_gee_available).items() if k != "points"},
    "get_drought_conditions": lambda: climate.drought(_gee_available),
    "rank_restoration_sites": lambda budget_usd=None: climate.rank_sites(budget_usd),
    "get_kelp_yearly": lambda county=None: _compact_yearly(climate.kelp_yearly(county)),
    "get_sea_temperature": lambda: _compact_sst(climate.kelp_yearly()),
    "get_enso_status": lambda: _compact_enso(climate.enso(_gee_available)),
    "get_data_quality": lambda: {"sources": [{"name": s["name"], "state": s["status"]["state"],
                                              "checks": s["status"]["checks"]} for s in climate.registry()]},
}
_client, _model = _llm()
AGENT = Agent(AGENT_TOOLS, _client, _model, KB,
              reasoning_effort=os.getenv("AGENT_REASONING_EFFORT", "low" if _model.startswith("openai/gpt-oss") else "") or None)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop.run_in_executor(executor, climate.refresh_kelp, _gee_available)
    loop.run_in_executor(executor, lambda: (climate.fire(_gee_available), climate.drought(_gee_available),
                                            climate.enso(_gee_available), climate.refresh_yearly(_gee_available)))
    yield

app = FastAPI(title="KelpWatch API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Status and data ───────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    k = climate.kelp()
    return {"gee_available": _gee_available, "kelp_status": k.get("status"),
            "kelp_computed_at": k.get("computed_at"), "county_count": len(k.get("counties", [])),
            "documents": KB.report["totals"]["documents"], "model": _model}

@app.get("/api/kelp")
def kelp_data(county: str | None = None):
    return climate.kelp(county)

@app.get("/api/wildfire")
def wildfire_data():
    return {k: v for k, v in climate.fire(_gee_available).items() if k != "points"}

@app.get("/api/wildfire/points")
def wildfire_points():
    d = climate.fire(_gee_available)
    return {"window": d.get("window"), "status": d.get("status"),
            "columns": ["lat", "lon", "peak_brightness_k", "region"], "points": d.get("points", [])}

@app.get("/api/drought")
def drought_data():
    return climate.drought(_gee_available)

@app.get("/api/kelp/yearly")
def kelp_yearly_data(county: str | None = None):
    return climate.kelp_yearly(county)

@app.get("/api/enso")
def enso_data():
    return climate.enso(_gee_available)

@app.get("/api/tiles/sst-anomaly")
def sst_anomaly_tiles():
    return _tile(climate.sst_anomaly_tile, fallback="sst")

@app.get("/api/sites")
def sites(budget_usd: float | None = None):
    return climate.rank_sites(budget_usd)

@app.get("/api/goal")
def goal():
    return {"target_acres": 10000, "target_year": 2040,
            "source": "RCW 79.135.440 (2022): WA DNR plan to conserve and restore at least 10,000 acres of kelp and eelgrass by 2040",
            "progress": None, "progress_note": "KelpWatch does not track restored acreage; see the WA DNR plan for progress."}

# ── Map tiles (Earth Engine) ──────────────────────────────────────────────────
def _tile(fn, *args, fallback: str | None = None):
    """Earth Engine tile URL; on failure, a labelled NASA GIBS fallback when one exists."""
    error = "Earth Engine not available"
    if _gee_available:
        try:
            return {"tile_url": fn(*args)}
        except Exception as e:
            error = str(e)[:200]
    if fallback:
        return {**climate.FALLBACK_TILES[fallback], "error": error}
    return {"tile_url": None, "error": error}

@app.get("/api/tiles/kelp/{period}")
def kelp_tiles(period: str):
    if period not in {p["label"] for p in climate.PERIODS}:
        raise HTTPException(404, "Unknown period")
    return _tile(climate.kelp_tile, period)

@app.get("/api/tiles/kelp-change")
def kelp_change_tiles():
    return _tile(climate.kelp_change_tile)

@app.get("/api/tiles/wildfire")
def wildfire_tiles():
    return _tile(climate.fire_tile)

@app.get("/api/tiles/drought")
def drought_tiles():
    return _tile(climate.drought_tile, fallback="drought")

# ── Transparency: sources, knowledge base, policy, skills, evals ──────────────
@app.get("/api/sources")
def sources():
    return {"sources": climate.registry(), "knowledge_base": KB.report}

@app.get("/api/kb/search")
def kb_search(q: str, k: int = 4):
    return {"query": q, "results": KB.search(q, k=min(k, 10))}

@app.get("/api/evals")
def evals():
    return {"retrieval": RETRIEVAL_EVAL}

@app.get("/api/policy")
def policy():
    return load_policy()

@app.get("/api/agent/config")
def agent_config():
    return {"model": _model, "provider": LLM_PROVIDER, "tools": TOOL_SPECS,
            "skills": [{"name": s["name"], "description": s["description"], "procedure": s["body"]}
                       for s in AGENT.skills.values()]}

# ── Agent (auth-gated, usage-limited) ─────────────────────────────────────────
class AgentQuery(BaseModel):
    query: str

@app.post("/api/agent")
def agent_query(req: AgentQuery, user: dict = Depends(verify_token)):
    user_id = user.get("sub", "local-dev")
    remaining = check_and_increment_usage(user_id)
    result = AGENT.run(req.query[:2000])
    return {**result, "queries_remaining": remaining}


# ── Usage endpoint ────────────────────────────────────────────────────────────
@app.get("/api/usage")
def get_usage(user: dict = Depends(verify_token)):
    user_id = user.get("sub", "local-dev")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or user_id == "local-dev":
        return {"queries_used": 0, "daily_limit": DAILY_QUERY_LIMIT, "queries_remaining": DAILY_QUERY_LIMIT}
    today = datetime.date.today().isoformat()
    hdrs = _sb_headers()
    r = http.get(
        f"{SUPABASE_URL}/rest/v1/token_usage", headers=hdrs,
        params={"user_id": f"eq.{user_id}", "query_date": f"eq.{today}", "select": "query_count"},
    )
    used = (r.json()[0]["query_count"] if r.ok and r.json() else 0)
    return {"queries_used": used, "daily_limit": DAILY_QUERY_LIMIT,
            "queries_remaining": max(0, DAILY_QUERY_LIMIT - used)}

# ── Contact endpoint ──────────────────────────────────────────────────────────
class ContactRequest(BaseModel):
    name: str
    email: str
    company: str = ""
    message: str

@app.post("/api/contact")
def submit_contact(req: ContactRequest):
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        hdrs = _sb_headers(json_body=True)
        http.post(
            f"{SUPABASE_URL}/rest/v1/contact_requests", headers=hdrs,
            json={"name": req.name, "email": req.email,
                  "company": req.company, "message": req.message},
        )
    return {"ok": True}

# ── Serve frontend ─────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory=pathlib.Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

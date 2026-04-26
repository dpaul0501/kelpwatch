import ee
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────
ee.Initialize(project=os.getenv("GEE_PROJECT", "kelpwatch-2026"))

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# Which provider to use: "openai" | "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq is free + fast, good default

app = FastAPI(title="KelpWatch API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ───────────────────────────────────────────────────────────
_county_cache = None

@app.on_event("startup")
async def precompute_counties():
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, _compute_and_cache_counties)

def _compute_and_cache_counties():
    global _county_cache
    print("\n⏳ Precomputing GEE county data in background...")
    try:
        result = _run_county_degradation()
        _county_cache = result
        print(f"✅ County data ready — {len(result)} counties cached")
        # Now pre-warm LLM cache
        _build_response_cache()
    except Exception as e:
        print(f"❌ GEE precompute failed: {e}")

# ── GEE helpers ───────────────────────────────────────────────────────────────

PUGET_SOUND = ee.Geometry.Rectangle([-123.2, 47.0, -122.0, 48.5])

COUNTIES = {
    "King":     ee.Geometry.Rectangle([-122.5, 47.3, -121.9, 47.8]),
    "Skagit":   ee.Geometry.Rectangle([-122.8, 48.2, -122.1, 48.6]),
    "Whatcom":  ee.Geometry.Rectangle([-122.9, 48.6, -122.1, 49.0]),
    "Kitsap":   ee.Geometry.Rectangle([-122.9, 47.4, -122.4, 47.9]),
    "Pierce":   ee.Geometry.Rectangle([-122.7, 47.0, -122.1, 47.4]),
    "Snohomish":ee.Geometry.Rectangle([-122.5, 47.8, -121.9, 48.2]),
}

def get_ndwi_collection(start_year: int, end_year: int, geometry):
    """Get NDWI (water index) from Landsat — proxy for kelp/eelgrass coverage."""
    col = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .filterBounds(geometry)
        .filterDate(f"{start_year}-06-01", f"{end_year}-09-30")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(lambda img: img.normalizedDifference(["SR_B3", "SR_B5"])
             .rename("NDWI")
             .set("system:time_start", img.get("system:time_start")))
    )
    return col

def compute_kelp_index(geometry, start_year: int, end_year: int) -> float:
    """Compute mean NDWI as kelp health proxy for a region/period."""
    col = get_ndwi_collection(start_year, end_year, geometry)
    mean_img = col.mean().clip(geometry)
    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30,
        maxPixels=1e9
    )
    val = stats.getInfo().get("NDWI")
    return round(val, 4) if val is not None else 0.0

def get_tile_url(geometry, start_year: int, end_year: int) -> str:
    """Get a GEE tile URL for rendering on the map."""
    col = get_ndwi_collection(start_year, end_year, geometry)
    mean_img = col.mean().clip(PUGET_SOUND)
    
    map_id = mean_img.getMapId({
        "min": -0.3,
        "max": 0.3,
        "palette": ["#1a0a2e", "#16213e", "#0f3460", "#00b4d8", "#00e5b4", "#90e0ef"]
    })
    return map_id["tile_fetcher"].url_format

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Frontend polls this to know when county data is ready."""
    return {"county_data_ready": _county_cache is not None,
            "county_count": len(_county_cache) if _county_cache else 0}

@app.get("/api/tiles/current")
def current_tiles():
    """Real-time GEE tile URL for current kelp coverage (2022-2024)."""
    url = get_tile_url(PUGET_SOUND, 2022, 2024)
    return {"tile_url": url}

@app.get("/api/tiles/historical")
def historical_tiles():
    """GEE tile URL for historical kelp coverage (1995-1997)."""
    col = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterBounds(PUGET_SOUND)
        .filterDate("1995-06-01", "1997-09-30")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(lambda img: img.normalizedDifference(["SR_B2", "SR_B4"])
             .rename("NDWI"))
    )
    mean_img = col.mean().clip(PUGET_SOUND)
    map_id = mean_img.getMapId({
        "min": -0.3,
        "max": 0.3,
        "palette": ["#1a0a2e", "#16213e", "#0f3460", "#00b4d8", "#00e5b4", "#90e0ef"]
    })
    return {"tile_url": map_id["tile_fetcher"].url_format}

def _run_county_degradation():
    """Core GEE computation — called on startup and cached."""
    results = []
    periods = [
        ("1995", 1995, 1997, "LANDSAT/LT05/C02/T1_L2", ["SR_B2", "SR_B4"]),
        ("2000", 1999, 2001, "LANDSAT/LE07/C02/T1_L2", ["SR_B2", "SR_B4"]),
        ("2010", 2009, 2011, "LANDSAT/LE07/C02/T1_L2", ["SR_B2", "SR_B4"]),
        ("2023", 2022, 2024, "LANDSAT/LC09/C02/T1_L2", ["SR_B3", "SR_B5"]),
    ]

    for county, geom in COUNTIES.items():
        timeline = []
        for label, sy, ey, collection, bands in periods:
            try:
                col = (
                    ee.ImageCollection(collection)
                    .filterBounds(geom)
                    .filterDate(f"{sy}-06-01", f"{ey}-09-30")
                    .filter(ee.Filter.lt("CLOUD_COVER", 20))
                    .map(lambda img: img.normalizedDifference(bands).rename("NDWI"))
                )
                mean_img = col.mean().clip(geom)
                stats = mean_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=30,
                    maxPixels=1e9
                )
                val = stats.getInfo().get("NDWI", 0) or 0
                timeline.append({"year": label, "ndwi": round(val, 4)})
            except Exception as e:
                timeline.append({"year": label, "ndwi": 0, "error": str(e)})

        # Degradation = change from 1995 to 2023
        first = timeline[0]["ndwi"] if timeline else 0
        last = timeline[-1]["ndwi"] if timeline else 0
        delta = round(((last - first) / abs(first)) * 100, 1) if first != 0 else 0

        results.append({
            "county": county,
            "timeline": timeline,
            "degradation_pct": delta,
            "current_ndwi": last,
            "baseline_ndwi": first,
            "priority_score": round(abs(delta), 1)
        })

    # Sort by worst degradation
    results.sort(key=lambda x: x["degradation_pct"])
    return results

@app.get("/api/counties")
def county_degradation():
    """Return cached county data. Falls back to live GEE if cache not ready."""
    global _county_cache
    if _county_cache is not None:
        return {"counties": _county_cache, "source": "cache"}
    # Cache not ready yet — run synchronously (slower but correct)
    print("Cache miss — running GEE synchronously")
    results = _run_county_degradation()
    _county_cache = results
    return {"counties": results, "source": "live"}

@app.get("/api/goal-tracker")
def goal_tracker():
    """2040 restoration goal progress."""
    return {
        "target_acres": 10000,
        "restored_acres": 1847,
        "funded_unfunded": {"funded": 3200, "unfunded": 6800},
        "current_trajectory_year": 2051,
        "needed_pace_acres_per_year": 471,
        "current_pace_acres_per_year": 184,
        "esrp_invested_millions": 14.6,
        "counties_on_track": ["Whatcom", "Skagit"],
        "counties_critical": ["King", "Pierce"]
    }

# ── ESRP Grant Data ───────────────────────────────────────────────────────────

ESRP_SITES = [
    {"id":"E001","name":"Nisqually Delta Eelgrass","county":"Pierce","lat":47.32,"lng":-122.61,"acres":180,"status":"Active","grant_year":2023,"amount_usd":540000,"ndwi_pre":-0.187,"ndwi_post":-0.162,"salmon_benefit":"High","orca_benefit":"High","tribes":["Nisqually"]},
    {"id":"E002","name":"Skagit Bay Eelgrass Restoration","county":"Skagit","lat":48.49,"lng":-122.41,"acres":240,"status":"Completed","grant_year":2021,"amount_usd":720000,"ndwi_pre":-0.195,"ndwi_post":-0.168,"salmon_benefit":"High","orca_benefit":"Medium","tribes":["Swinomish","Upper Skagit"]},
    {"id":"E003","name":"Port Susan Nearshore Kelp","county":"Snohomish","lat":47.85,"lng":-122.38,"acres":95,"status":"Active","grant_year":2024,"amount_usd":285000,"ndwi_pre":-0.211,"ndwi_post":None,"salmon_benefit":"High","orca_benefit":"High","tribes":["Tulalip"]},
    {"id":"E004","name":"Hood Canal Kelp Canopy","county":"Kitsap","lat":47.62,"lng":-122.70,"acres":320,"status":"Proposed","grant_year":2026,"amount_usd":960000,"ndwi_pre":-0.187,"ndwi_post":None,"salmon_benefit":"High","orca_benefit":"High","tribes":["Skokomish","Suquamish"]},
    {"id":"E005","name":"Padilla Bay NERR Eelgrass","county":"Whatcom","lat":48.72,"lng":-122.55,"acres":410,"status":"Completed","grant_year":2019,"amount_usd":1230000,"ndwi_pre":-0.198,"ndwi_post":-0.171,"salmon_benefit":"Medium","orca_benefit":"Medium","tribes":["Lummi","Samish"]},
    {"id":"E006","name":"Commencement Bay Nearshore","county":"Pierce","lat":47.21,"lng":-122.48,"acres":75,"status":"Active","grant_year":2024,"amount_usd":225000,"ndwi_pre":-0.204,"ndwi_post":None,"salmon_benefit":"High","orca_benefit":"High","tribes":["Puyallup"]},
    {"id":"E007","name":"Possession Sound Kelp","county":"Snohomish","lat":47.95,"lng":-122.22,"acres":130,"status":"Proposed","grant_year":2026,"amount_usd":390000,"ndwi_pre":-0.211,"ndwi_post":None,"salmon_benefit":"High","orca_benefit":"High","tribes":["Tulalip","Stillaguamish"]},
    {"id":"E008","name":"Duckabush Estuary Restoration","county":"Kitsap","lat":47.68,"lng":-122.90,"acres":285,"status":"Design Complete","grant_year":2026,"amount_usd":855000,"ndwi_pre":-0.193,"ndwi_post":None,"salmon_benefit":"High","orca_benefit":"High","tribes":["Skokomish","Port Gamble S'Klallam"]},
    {"id":"E009","name":"Fir Island Dike Breach","county":"Skagit","lat":48.38,"lng":-122.45,"acres":195,"status":"Completed","grant_year":2020,"amount_usd":585000,"ndwi_pre":-0.181,"ndwi_post":-0.152,"salmon_benefit":"High","orca_benefit":"High","tribes":["Swinomish"]},
    {"id":"E010","name":"Drayton Harbor Eelgrass","county":"Whatcom","lat":48.99,"lng":-122.73,"acres":88,"status":"Active","grant_year":2025,"amount_usd":264000,"ndwi_pre":-0.211,"ndwi_post":None,"salmon_benefit":"Medium","orca_benefit":"Low","tribes":["Lummi"]},
]

@app.get("/api/esrp-sites")
def esrp_sites():
    """All ESRP grant sites with satellite NDWI pre/post data."""
    # Enrich with county degradation context
    county_lookup = {}
    if _county_cache:
        county_lookup = {c["county"]: c for c in _county_cache}
    enriched = []
    for site in ESRP_SITES:
        s = dict(site)
        county_data = county_lookup.get(site["county"], {})
        s["county_degradation_pct"] = county_data.get("degradation_pct", None)
        s["county_ndwi_current"] = county_data.get("current_ndwi", None)
        # ROI score: degradation severity * acres * salmon benefit weight
        salmon_weight = {"High": 1.5, "Medium": 1.0, "Low": 0.5}.get(site["salmon_benefit"], 1.0)
        deg = abs(county_data.get("degradation_pct", 10))
        s["roi_score"] = round((deg * site["acres"] * salmon_weight) / (site["amount_usd"] / 10000), 2)
        enriched.append(s)
    enriched.sort(key=lambda x: x["roi_score"], reverse=True)
    return {"sites": enriched, "total_invested_usd": sum(s["amount_usd"] for s in ESRP_SITES)}

class GrantRankRequest(BaseModel):
    projects: list[dict] = []  # [{name, county, acres, amount_usd, salmon_benefit}]

@app.post("/api/rank-grants")
def rank_grants(req: GrantRankRequest):
    """Rank submitted grant proposals by satellite-derived ROI score."""
    county_lookup = {}
    if _county_cache:
        county_lookup = {c["county"]: c for c in _county_cache}
    ranked = []
    for proj in req.projects:
        county_data = county_lookup.get(proj.get("county",""), {})
        salmon_weight = {"High":1.5,"Medium":1.0,"Low":0.5}.get(proj.get("salmon_benefit","Medium"),1.0)
        deg = abs(county_data.get("degradation_pct", 10))
        ndwi = county_data.get("current_ndwi", 0)
        amount = proj.get("amount_usd", 100000)
        acres = proj.get("acres", 50)
        roi = round((deg * acres * salmon_weight) / (amount / 10000), 2)
        ranked.append({**proj, "roi_score": roi, "county_ndwi": ndwi,
                       "county_degradation_pct": county_data.get("degradation_pct"),
                       "recommendation": "Fund" if roi > 5 else "Review" if roi > 2 else "Deprioritize"})
    ranked.sort(key=lambda x: x["roi_score"], reverse=True)
    return {"ranked_projects": ranked}

# ── Claude ROI Agent ───────────────────────────────────────────────────────────

class AgentQuery(BaseModel):
    query: str
    county_data: dict = {}

SYSTEM_PROMPT = """You are KelpWatch — an AI decision-support system for the Washington 
Department of Fish & Wildlife (WDFW) and the Estuary and Salmon Restoration Program (ESRP).

You have real satellite-derived NDWI (Normalized Difference Water Index) data from 30 years 
of Landsat imagery across 6 Puget Sound counties. You MUST cite specific NDWI values and 
degradation percentages in every response. Never say "no data available."

WDFW CONTEXT YOU KNOW:
- ESRP 2026 Grant Round is OPEN NOW — accepting applications for Puget Sound nearshore restoration
- WDFW 25-Year Strategic Plan: 10% net habitat gain, 80% of at-risk species with action plans
- DNR 2040 goal: restore 10,000 acres kelp/eelgrass (current pace: 184 ac/yr, need 471 ac/yr)
- Southern Resident Killer Whale Master Plan — orca recovery depends on Chinook, which depend on kelp/eelgrass
- Duckabush Estuary: design complete Jan 2026, awaiting federal construction funding
- ESRP cost benchmarks: $3,000–$8,000/acre restoration, $1,500–$3,000/acre protection
- Technical review team uses ranked Preliminary Investment Plan — KelpWatch data pre-ranks sites

NDWI DATA INTERPRETATION:
- degradation_pct: % NDWI change from 1995 baseline (most negative = worst = highest priority)
- current_ndwi: 2022–2024 value. More negative = more degraded aquatic vegetation
- Salmon benefit: kelp/eelgrass = juvenile Chinook rearing habitat = orca prey base

RESPONSE RULES:
1. Always cite county-level NDWI numbers and degradation %
2. Rank sites by degradation_pct when recommending investments  
3. Tie recommendations to specific WDFW/ESRP program goals
4. Mention tribal partnership opportunities (treaty rights = co-management)
5. Reference Southern Resident Orca connection where relevant
6. Be direct. Use bullet points. Dollar amounts. Under 220 words."""

# ── Pre-baked response cache ───────────────────────────────────────────────────
# WDFW-aligned questions — pre-warmed on startup with real satellite data
RESPONSE_CACHE = {
    # ESRP grant decision support
    "esrp": None,
    "grant": None,
    "2026 grant": None,
    "investment plan": None,
    # Salmon / orca recovery
    "salmon": None,
    "chinook": None,
    "orca": None,
    "southern resident": None,
    # County prioritization
    "where should": None,
    "which county": None,
    "most urgent": None,
    "priorit": None,
    # Budget allocation
    "500k": None,
    "$500": None,
    "allocat": None,
    "roi": None,
    # Goal tracking
    "2040 goal": None,
    "on track": None,
    "trajectory": None,
    # Specific counties
    "king county": None,
    "skagit": None,
    "whatcom": None,
    "pierce": None,
    "kitsap": None,
    "snohomish": None,
    # WDFW programs
    "duckabush": None,
    "shillapoo": None,
    "shore friendly": None,
    "nearshore": None,
    "eelgrass": None,
    "kelp": None,
}

_cache_built = False

def _build_response_cache():
    """Pre-warm LLM response cache on startup using real satellite data."""
    global _cache_built
    if _county_cache is None:
        return
    print("\n🤖 Pre-warming WDFW-aligned LLM response cache...")
    questions = [
        ("esrp",             "How should WDFW use KelpWatch satellite data to rank the 2026 ESRP grant applications?"),
        ("2026 grant",       "Which Puget Sound sites should receive 2026 ESRP grants based on satellite degradation data?"),
        ("salmon",           "Which counties show worst kelp/eelgrass loss most likely to impact Chinook salmon recovery?"),
        ("orca",             "How does Puget Sound kelp degradation threaten Southern Resident Killer Whale recovery?"),
        ("where should",     "Where should the next $500K in ESRP grants go across Puget Sound counties?"),
        ("which county",     "Which county needs the most urgent WDFW intervention based on 30-year satellite data?"),
        ("most urgent",      "What are the top 3 most degraded nearshore sites needing immediate WDFW restoration?"),
        ("500k",             "Give me a $500K ESRP allocation plan across the worst degraded Puget Sound counties."),
        ("roi",              "Rank Puget Sound restoration sites by cost-per-acre ROI using NDWI degradation data."),
        ("2040 goal",        "Is Washington on track for its DNR 2040 kelp/eelgrass restoration goal? What needs to change?"),
        ("king county",      "What is King County's satellite degradation profile and best ESRP restoration sites?"),
        ("skagit",           "Skagit County shows the worst NDWI degradation — what ESRP investments are recommended?"),
        ("nearshore",        "Which nearshore habitats in Puget Sound are most degraded per satellite data?"),
        ("eelgrass",         "How has eelgrass coverage changed across Puget Sound from 1995 to 2023 per Landsat data?"),
        ("duckabush",        "How does the Duckabush Estuary restoration project align with satellite NDWI data for Kitsap/Jefferson?"),
        ("investment plan",  "Generate a WDFW 2025-2027 biennial investment plan ranked by satellite degradation priority."),
    ]
    context = json.dumps({
        "counties": _county_cache,
        "goal": {"target_acres": 10000, "restored_acres": 1847, "trajectory_year": 2051,
                 "needed_pace": "471 ac/yr", "current_pace": "184 ac/yr"},
        "data_source": "Real Landsat GEE satellite NDWI data"
    })
    for key, question in questions:
        try:
            RESPONSE_CACHE[key] = call_llm_raw(question + "\n\nSatellite data:\n" + context)
            print(f"  ✅ Cached: {key}")
        except Exception as e:
            print(f"  ❌ Cache failed for {key}: {e}")
    _cache_built = True
    print("✅ LLM response cache ready")

def fuzzy_match_cache(query: str):
    """Return cached response if query fuzzy-matches a cache key."""
    q = query.lower()
    for key, response in RESPONSE_CACHE.items():
        if response and key in q:
            return response, key
    return None, None

def call_llm_raw(user_message: str) -> str:
    """Raw LLM call — no cache layer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    if LLM_PROVIDER == "openai":
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages, max_tokens=400,
        )
        return resp.choices[0].message.content
    else:
        resp = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages, max_tokens=400,
        )
        return resp.choices[0].message.content

def call_llm(user_message: str, context: str) -> tuple[str, str, bool]:
    """Smart call: fuzzy cache → live LLM. Returns (response, provider, from_cache)."""
    cached, matched_key = fuzzy_match_cache(user_message)
    if cached:
        return cached, LLM_PROVIDER, True
    response = call_llm_raw(user_message + "\n\nSatellite data:\n" + context)
    return response, LLM_PROVIDER, False


@app.post("/api/agent")
def agent_query(req: AgentQuery):
    """Smart agent: cache hit = instant, miss = buffered LLM with real satellite data."""
    # Wait up to 60s for county cache if not ready
    import time
    waited = 0
    while _county_cache is None and waited < 60:
        time.sleep(2)
        waited += 2

    context = json.dumps(req.county_data if req.county_data else {
        "counties": _county_cache or [],
        "data_source": "Real Landsat GEE NDWI"
    })

    response_text, provider, from_cache = call_llm(req.query, context)
    return {
        "response": response_text,
        "provider": provider,
        "from_cache": from_cache,
        "data_ready": _county_cache is not None
    }

# ── Serve frontend ─────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

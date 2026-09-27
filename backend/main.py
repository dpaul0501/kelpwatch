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
    if not _gee_sa_key and _gee_key_file and os.path.exists(_gee_key_file):
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
    print(f"⚠ GEE unavailable — kelp tiles degraded. Fix: set GEE_CREDENTIALS_JSON on Render.\n  {_gee_err}")

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

# ── In-memory cache ───────────────────────────────────────────────────────────
_county_cache = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, _compute_and_cache_counties)
    yield

app = FastAPI(title="KelpWatch API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _compute_and_cache_counties():
    global _county_cache
    print("\n⏳ Precomputing GEE county data in background...")
    try:
        result = _run_county_degradation()
        _county_cache = result
        print(f"✅ County data ready — {len(result)} counties cached")
        _build_response_cache()
    except Exception as e:
        print(f"❌ GEE precompute failed: {e}")

# ── GEE constants ─────────────────────────────────────────────────────────────
try:
    PUGET_SOUND  = ee.Geometry.Rectangle([-123.2, 47.0, -122.0, 48.5])
    INDIA_BOUNDS = ee.Geometry.Rectangle([68.0, 6.0, 97.5, 36.0])
    US_BOUNDS    = ee.Geometry.Rectangle([-125.0, 24.0, -66.0, 49.0])
    COUNTIES = {
        "King":      ee.Geometry.Rectangle([-122.5, 47.3, -121.9, 47.8]),
        "Skagit":    ee.Geometry.Rectangle([-122.8, 48.2, -122.1, 48.6]),
        "Whatcom":   ee.Geometry.Rectangle([-122.9, 48.6, -122.1, 49.0]),
        "Kitsap":    ee.Geometry.Rectangle([-122.9, 47.4, -122.4, 47.9]),
        "Pierce":    ee.Geometry.Rectangle([-122.7, 47.0, -122.1, 47.4]),
        "Snohomish": ee.Geometry.Rectangle([-122.5, 47.8, -121.9, 48.2]),
    }
except Exception:
    PUGET_SOUND = INDIA_BOUNDS = US_BOUNDS = None
    COUNTIES = {}

# ── GEE helpers ───────────────────────────────────────────────────────────────
def get_ndwi_collection(start_year: int, end_year: int, geometry):
    return (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"))
        .filterBounds(geometry)
        .filterDate(f"{start_year}-06-01", f"{end_year}-09-30")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(lambda img: img.normalizedDifference(["SR_B3", "SR_B5"])
             .rename("NDWI")
             .set("system:time_start", img.get("system:time_start")))
    )

def get_tile_url(geometry, start_year: int, end_year: int) -> str:
    col = get_ndwi_collection(start_year, end_year, geometry)
    mean_img = col.mean().clip(PUGET_SOUND)
    map_id = mean_img.getMapId({
        "min": -0.3, "max": 0.3,
        "palette": ["#1a0a2e", "#16213e", "#0f3460", "#00b4d8", "#00e5b4", "#90e0ef"]
    })
    return map_id["tile_fetcher"].url_format

# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {"county_data_ready": _county_cache is not None,
            "county_count": len(_county_cache) if _county_cache else 0}

# ── Kelp tiles ────────────────────────────────────────────────────────────────
@app.get("/api/tiles/current")
def current_tiles():
    if not _gee_available or PUGET_SOUND is None:
        return {"tile_url": None, "gee_available": False}
    try:
        return {"tile_url": get_tile_url(PUGET_SOUND, 2022, 2024), "gee_available": True}
    except Exception as e:
        return {"tile_url": None, "gee_available": False, "error": str(e)}

@app.get("/api/tiles/historical")
def historical_tiles():
    if not _gee_available or PUGET_SOUND is None:
        return {"tile_url": None, "gee_available": False}
    try:
        col = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterBounds(PUGET_SOUND)
            .filterDate("1995-06-01", "1997-09-30")
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .map(lambda img: img.normalizedDifference(["SR_B2", "SR_B4"]).rename("NDWI"))
        )
        mean_img = col.mean().clip(PUGET_SOUND)
        map_id = mean_img.getMapId({
            "min": -0.3, "max": 0.3,
            "palette": ["#1a0a2e", "#16213e", "#0f3460", "#00b4d8", "#00e5b4", "#90e0ef"]
        })
        return {"tile_url": map_id["tile_fetcher"].url_format}
    except Exception as e:
        return {"tile_url": None, "gee_available": False, "error": str(e)}

# ── NASA GIBS tile helper (no auth, free) ─────────────────────────────────────
def _gibs_url(layer: str, matrix_set: str, fmt: str, days_ago: int = 1) -> str:
    d = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
    return f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{d}/{matrix_set}/{{z}}/{{y}}/{{x}}.{fmt}"

# ── Forest Fire tiles — NASA GIBS VIIRS (no GEE needed) ──────────────────────
@app.get("/api/tiles/fires")
def fire_tiles():
    """NASA GIBS VIIRS active fire tile — updated daily, no GEE."""
    return {"tile_url": _gibs_url("VIIRS_SNPP_Fires_All", "GoogleMapsCompatible_Level6", "png", days_ago=1)}

def _firms_fetch(bbox: str, days: int = 5) -> list[dict]:
    """Fetch VIIRS SNPP NRT CSV from FIRMS and return parsed rows."""
    key = os.getenv("FIRMS_MAP_KEY", "")
    if not key:
        return []
    r = http.get(
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bbox}/{days}",
        timeout=15,
    )
    if not r.ok:
        return []
    lines = r.text.strip().split("\n")
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        rows.append(row)
    return rows

@app.get("/api/fire/hotspots")
def fire_hotspots():
    """NASA FIRMS VIIRS hotspots as GeoJSON — last 7 days, US + India."""
    regions = [("US", "-130,24,-60,50"), ("India", "68,6,97,36")]
    features = []
    for region, bbox in regions:
        for row in _firms_fetch(bbox):
            try:
                conf = row.get("confidence", "n")
                if conf == "l":          # skip low-confidence detections
                    continue
                frp = float(row.get("frp", 0) or 0)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [
                        float(row["longitude"]), float(row["latitude"])
                    ]},
                    "properties": {
                        "region": region,
                        "confidence": conf,   # h=high, n=nominal
                        "frp": round(frp, 1), # fire radiative power MW
                        "date": row.get("acq_date", ""),
                        "daynight": row.get("daynight", ""),
                    },
                })
            except (ValueError, KeyError):
                continue
    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features),
        "source": "NASA FIRMS VIIRS SNPP NRT · 375m · last 7 days",
    }

@app.get("/api/fire/summary")
def fire_summary():
    """NASA FIRMS fire counts + risk context for US and India."""
    us_rows    = _firms_fetch("-130,24,-60,50")
    india_rows = _firms_fetch("68,6,97,36")

    def _count(rows: list[dict]) -> dict:
        high = sum(1 for r in rows if r.get("confidence") == "h")
        nom  = sum(1 for r in rows if r.get("confidence") == "n")
        frp_vals = [float(r["frp"]) for r in rows if r.get("frp") and r["frp"].strip()]
        return {
            "total_hotspots_7d": len(rows),
            "high_confidence": high,
            "nominal_confidence": nom,
            "max_frp_mw": round(max(frp_vals), 1) if frp_vals else None,
            "avg_frp_mw": round(sum(frp_vals) / len(frp_vals), 1) if frp_vals else None,
        }

    has_firms = bool(os.getenv("FIRMS_MAP_KEY"))
    return {
        "us": {
            **_count(us_rows),
            "high_risk_states": ["California", "Oregon", "Washington", "Colorado", "Idaho"],
            "season_status": "Active fire season May–October",
        },
        "india": {
            **_count(india_rows),
            "high_risk_regions": ["Uttarakhand", "Himachal Pradesh", "Odisha", "Chhattisgarh"],
            "season_status": "Pre-monsoon fire risk March–May; fires year-round in central India",
        },
        "data_source": "NASA FIRMS VIIRS SNPP NRT · 375m" if has_firms else "FIRMS key not configured",
        "data_period": "Last 5 days (VIIRS NRT)",
    }

# ── India Monsoon / Farmer Risk — Open-Meteo (no GEE needed) ─────────────────
@app.get("/api/tiles/india-rainfall")
def india_rainfall_tiles():
    """NASA GIBS GPM precipitation tile over India — no GEE."""
    return {"tile_url": _gibs_url("GPM_L3_Half_Hourly_04_precipitation", "GoogleMapsCompatible_Level6", "png", days_ago=2)}

@app.get("/api/tiles/india-ndvi")
def india_ndvi_tiles():
    """NASA GIBS MODIS NDVI tile over India — no GEE."""
    return {"tile_url": _gibs_url("MODIS_Terra_Land_Surface_Temp_Day", "GoogleMapsCompatible_Level7", "png", days_ago=2)}

@app.get("/api/tiles/sst")
def sst_tiles():
    """NASA GIBS MUR SST tile — Pacific + Indian Ocean, El Niño visualization."""
    return {"tile_url": _gibs_url("MUR-JPL-L4-GLOB-v4.1_analysed_sst", "GoogleMapsCompatible_Level7", "png", days_ago=3)}

@app.get("/api/india/summary")
def india_climate_summary():
    """India rainfall from Open-Meteo (no GEE) + farmer risk assessment."""
    cities = [
        ("Mumbai",    19.076,  72.878),
        ("Delhi",     28.614,  77.209),
        ("Chennai",   13.083,  80.271),
        ("Kolkata",   22.573,  88.364),
        ("Bhopal",    23.260,  77.413),
        ("Hyderabad", 17.385,  78.487),
        ("Jaipur",    26.912,  75.787),
        ("Patna",     25.594,  85.138),
    ]
    rain_values = []
    city_data = []
    for name, lat, lon in cities:
        try:
            r = http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": "precipitation_sum",
                    "past_days": 30, "forecast_days": 1,
                    "timezone": "Asia/Kolkata",
                },
                timeout=6,
            )
            if r.ok:
                vals = r.json()["daily"]["precipitation_sum"]
                city_mm = round(sum(v or 0 for v in vals), 1)
                rain_values.append(city_mm)
                city_data.append({"name": name, "lat": lat, "lon": lon, "rainfall_mm": city_mm})
        except Exception:
            pass

    rain_mm = round(sum(rain_values) / len(rain_values), 1) if rain_values else None
    drought_risk = "High" if rain_mm is not None and rain_mm < 50 else (
                   "Moderate" if rain_mm is not None and rain_mm < 150 else "Low")
    crop_stress  = "Moderate"  # conservative default without live NDVI

    return {
        "rainfall_mm_30d": rain_mm,
        "ndvi_mean": None,
        "drought_risk": drought_risk,
        "crop_stress": crop_stress,
        "farmer_advisory": (
            "⚠ Drought alert — below-normal rainfall. Kharif sowing at risk."
            if drought_risk == "High" else
            "Rainfall near-normal. Monitor for late-season deficit."
            if drought_risk == "Moderate" else
            "Adequate rainfall. Normal Kharif/Rabi planning."
        ),
        "city_count": len(rain_values),
        "cities": city_data,
        "notes": f"8-city rainfall average · Open-Meteo · 30-day accumulated",
        "source": "Open-Meteo",
    }

@app.get("/api/elnino")
def elnino_status():
    """NOAA Oceanic Niño Index (ONI) — live El Niño / La Niña status."""
    try:
        r = http.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=10)
        lines = [l for l in r.text.strip().split("\n") if l.strip() and not l.startswith("SEAS")]
        oni = float(lines[-1].split()[-1])
        status = "El Niño" if oni >= 0.5 else "La Niña" if oni <= -0.5 else "Neutral"
        impact = (
            "Below-normal monsoon onset likely · elevated Kharif drought risk"
            if oni >= 0.5 else
            "Above-normal monsoon likely · La Niña boost for India"
            if oni <= -0.5 else
            "Normal monsoon conditions expected"
        )
        return {"oni": oni, "status": status, "impact_india": impact,
                "source": "NOAA CPC ONI", "live": True}
    except Exception:
        return {"oni": None, "status": "Unknown", "impact_india": "NOAA data unavailable",
                "source": "NOAA CPC ONI", "live": False}

# ── County / kelp data ────────────────────────────────────────────────────────
def _run_county_degradation():
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
                val = col.mean().clip(geom).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9
                ).getInfo().get("NDWI", 0) or 0
                timeline.append({"year": label, "ndwi": round(val, 4)})
            except Exception as e:
                timeline.append({"year": label, "ndwi": 0, "error": str(e)})

        first = timeline[0]["ndwi"] if timeline else 0
        last  = timeline[-1]["ndwi"] if timeline else 0
        delta = round(((last - first) / abs(first)) * 100, 1) if first != 0 else 0
        results.append({
            "county": county,
            "timeline": timeline,
            "degradation_pct": delta,
            "current_ndwi": last,
            "baseline_ndwi": first,
            "priority_score": round(abs(delta), 1)
        })
    results.sort(key=lambda x: x["degradation_pct"])
    return results

@app.get("/api/counties")
def county_degradation():
    global _county_cache
    if _county_cache is not None:
        return {"counties": _county_cache, "source": "cache"}
    print("Cache miss — running GEE synchronously")
    results = _run_county_degradation()
    _county_cache = results
    return {"counties": results, "source": "live"}

@app.get("/api/goal-tracker")
def goal_tracker():
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

# ── ESRP grant data ───────────────────────────────────────────────────────────
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
    county_lookup = {c["county"]: c for c in _county_cache} if _county_cache else {}
    enriched = []
    for site in ESRP_SITES:
        s = dict(site)
        cd = county_lookup.get(site["county"], {})
        s["county_degradation_pct"] = cd.get("degradation_pct")
        s["county_ndwi_current"]    = cd.get("current_ndwi")
        salmon_weight = {"High": 1.5, "Medium": 1.0, "Low": 0.5}.get(site["salmon_benefit"], 1.0)
        deg = abs(cd.get("degradation_pct", 10))
        s["roi_score"] = round((deg * site["acres"] * salmon_weight) / (site["amount_usd"] / 10000), 2)
        enriched.append(s)
    enriched.sort(key=lambda x: x["roi_score"], reverse=True)
    return {"sites": enriched, "total_invested_usd": sum(s["amount_usd"] for s in ESRP_SITES)}

class GrantRankRequest(BaseModel):
    projects: list[dict] = []

@app.post("/api/rank-grants")
def rank_grants(req: GrantRankRequest):
    county_lookup = {c["county"]: c for c in _county_cache} if _county_cache else {}
    ranked = []
    for proj in req.projects:
        cd = county_lookup.get(proj.get("county", ""), {})
        sw  = {"High": 1.5, "Medium": 1.0, "Low": 0.5}.get(proj.get("salmon_benefit", "Medium"), 1.0)
        deg = abs(cd.get("degradation_pct", 10))
        roi = round((deg * proj.get("acres", 50) * sw) / (proj.get("amount_usd", 100000) / 10000), 2)
        ranked.append({
            **proj,
            "roi_score": roi,
            "county_ndwi": cd.get("current_ndwi"),
            "county_degradation_pct": cd.get("degradation_pct"),
            "recommendation": "Fund" if roi > 5 else "Review" if roi > 2 else "Deprioritize"
        })
    ranked.sort(key=lambda x: x["roi_score"], reverse=True)
    return {"ranked_projects": ranked}

# ── Intent routing ────────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "kelp": [
        "kelp", "eelgrass", "ndwi", "puget", "wdfw", "esrp", "salmon", "chinook",
        "orca", "killer whale", "restoration", "nearshore", "grant", "degradation",
        "habitat", "2040", "dnr", "duckabush", "skagit", "whatcom", "kitsap",
        "pierce", "snohomish", "king county", "seagrass", "landsat", "marine",
        "aquatic vegetation", "puget sound", "biennial", "investment plan"
    ],
    "fire": [
        "fire", "wildfire", "forest fire", "burn", "flame", "smoke", "blaze",
        "modis", "viirs", "burnt area", "fire risk", "fire season", "fire damage",
        "uttarakhand", "himachal", "california fire", "oregon fire", "wildfire risk",
        "fire monitoring", "fire detection", "fire alert"
    ],
    "india": [
        "india", "monsoon", "rainfall", "drought", "farmer", "crop", "kharif",
        "rabi", "chirps", "maharashtra", "rajasthan", "gujarat", "uttar pradesh",
        "madhya pradesh", "bihar", "punjab", "haryana", "monsoon deficit",
        "flood risk", "india climate", "indian agriculture", "india rain",
        "precipitation", "agricultural risk", "crop stress", "india ndvi"
    ]
}

# Signals that a query is definitively unrelated to any supported domain
_OUT_OF_SCOPE = [
    "recipe", "cook", "bake", "restaurant", "movie", "song", "write code",
    "programming", "javascript", "python code", "sql query", "joke",
    "poem", "story", "translate", "stock price", "crypto", "bitcoin",
    "medical advice", "diagnosis", "legal advice", "tax advice"
]

def classify_intent(query: str) -> str:
    """Route query to kelp / fire / india / out_of_scope."""
    q = query.lower()
    if any(sig in q for sig in _OUT_OF_SCOPE):
        return "out_of_scope"
    scores = {
        topic: sum(1 for kw in kws if kw in q)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "kelp"  # default to kelp when ambiguous

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a multi-domain satellite climate intelligence agent with three specializations:

1. KELP & EELGRASS RESTORATION — Puget Sound, WA
   - 30-year Landsat NDWI data across 6 counties (King, Skagit, Whatcom, Kitsap, Pierce, Snohomish)
   - ESRP 2026 grant prioritization, WDFW decision support, DNR 2040 goal (10,000 acres)
   - Southern Resident Killer Whale recovery via Chinook salmon habitat
   - Cost benchmarks: $3,000–$8,000/acre restoration, $1,500–$3,000/acre protection

2. FOREST FIRE RISK — US & India
   - MODIS MCD64A1 burned area data · VIIRS active fire detection
   - US: California, Oregon, Washington, Colorado, Idaho high risk (May–Oct season)
   - India: Uttarakhand, Himachal Pradesh, Odisha, Chhattisgarh (pre-monsoon March–May)
   - Cite burned area in km² or pixel counts when available

3. INDIA MONSOON & FARMER RISK
   - CHIRPS daily rainfall · MODIS NDVI vegetation health over India
   - Kharif (June–Sept sowing) and Rabi (Oct–March) season risk
   - NDVI < 3000 (scaled) = severe crop stress · rainfall < 50mm/30d = drought alert
   - State-level risks: Maharashtra, Rajasthan, MP, Gujarat, UP, Bihar, Punjab, Haryana

RESPONSE RULES:
1. Cite specific satellite numbers — NDWI, rainfall mm, burn area km², NDVI values
2. Connect data to real-world impact (farmers, ecosystems, communities, policy)
3. Give actionable recommendations — dollar amounts, percentages, season timelines
4. Use bullet points. Under 260 words. Direct language.
5. For out-of-domain questions, say so briefly and redirect to supported topics."""

# ── Response cache ────────────────────────────────────────────────────────────
RESPONSE_CACHE: dict[str, str | None] = {
    # Kelp
    "esrp": None, "grant": None, "2026 grant": None, "investment plan": None,
    "salmon": None, "chinook": None, "orca": None, "southern resident": None,
    "where should": None, "which county": None, "most urgent": None,
    "500k": None, "roi": None, "2040 goal": None, "on track": None,
    "king county": None, "skagit": None, "whatcom": None, "pierce": None,
    "kitsap": None, "snohomish": None, "duckabush": None,
    "nearshore": None, "eelgrass": None, "kelp": None,
    # Fire
    "wildfire": None, "forest fire": None, "fire risk": None,
    "california fire": None, "india fire": None, "fire season": None,
    # India
    "monsoon": None, "india rainfall": None, "india drought": None,
    "farmer risk": None, "kharif": None, "india ndvi": None,
}

_cache_built = False

def _build_response_cache():
    global _cache_built
    if _county_cache is None:
        return
    print("\n🤖 Pre-warming LLM response cache...")
    kelp_ctx = json.dumps({
        "counties": _county_cache,
        "goal": {"target_acres": 10000, "restored_acres": 1847, "trajectory_year": 2051,
                 "needed_pace": "471 ac/yr", "current_pace": "184 ac/yr"},
        "data_source": "Real Landsat GEE satellite NDWI"
    })
    fire_ctx = json.dumps({
        "fire": {"us": {"high_risk_states": ["California","Oregon","Washington","Colorado","Idaho"],
                        "season_status": "Active May–Oct"},
                 "india": {"high_risk_regions": ["Uttarakhand","Himachal Pradesh","Odisha","Chhattisgarh"],
                           "season_status": "Pre-monsoon March–May"}},
        "data_source": "MODIS MCD64A1"
    })
    india_ctx = json.dumps({
        "india_climate": {"drought_risk": "Moderate", "crop_stress": "Moderate",
                          "farmer_advisory": "Monitor rainfall deficit for Kharif season"},
        "data_source": "CHIRPS/MODIS India"
    })
    questions = [
        ("esrp",         kelp_ctx, "How should WDFW use KelpWatch satellite data to rank the 2026 ESRP grant applications?"),
        ("2026 grant",   kelp_ctx, "Which Puget Sound sites should receive 2026 ESRP grants based on satellite degradation data?"),
        ("salmon",       kelp_ctx, "Which counties show worst kelp/eelgrass loss most likely to impact Chinook salmon recovery?"),
        ("orca",         kelp_ctx, "How does Puget Sound kelp degradation threaten Southern Resident Killer Whale recovery?"),
        ("where should", kelp_ctx, "Where should the next $500K in ESRP grants go across Puget Sound counties?"),
        ("which county", kelp_ctx, "Which county needs the most urgent WDFW intervention based on 30-year satellite data?"),
        ("500k",         kelp_ctx, "Give me a $500K ESRP allocation plan across the worst degraded Puget Sound counties."),
        ("roi",          kelp_ctx, "Rank Puget Sound restoration sites by cost-per-acre ROI using NDWI degradation data."),
        ("2040 goal",    kelp_ctx, "Is Washington on track for its DNR 2040 kelp/eelgrass restoration goal?"),
        ("king county",  kelp_ctx, "What is King County's satellite degradation profile and best ESRP restoration sites?"),
        ("skagit",       kelp_ctx, "Skagit County shows worst NDWI degradation — what ESRP investments are recommended?"),
        ("nearshore",    kelp_ctx, "Which nearshore habitats in Puget Sound are most degraded per satellite data?"),
        ("duckabush",    kelp_ctx, "How does Duckabush Estuary restoration align with satellite NDWI data for Kitsap?"),
        ("wildfire",     fire_ctx, "What are the current highest wildfire risk areas in the US based on MODIS data?"),
        ("india fire",   fire_ctx, "Which Indian states face the highest forest fire risk and when is peak season?"),
        ("fire risk",    fire_ctx, "Compare forest fire risk between the US West Coast and India using satellite data."),
        ("monsoon",      india_ctx,"What is the current monsoon rainfall status across India and its impact on farmers?"),
        ("india drought",india_ctx,"Which Indian states face drought risk this Kharif season based on CHIRPS rainfall data?"),
        ("farmer risk",  india_ctx,"How should Indian farmers in drought-prone states plan for below-normal monsoon rainfall?"),
        ("kharif",       india_ctx,"What does satellite rainfall data say about Kharif crop prospects in major farming states?"),
    ]
    for key, ctx, question in questions:
        try:
            RESPONSE_CACHE[key] = call_llm_raw(question + "\n\nSatellite data:\n" + ctx)
            print(f"  ✅ Cached: {key}")
        except Exception as e:
            print(f"  ❌ Cache failed for {key}: {e}")
    _cache_built = True
    print("✅ LLM cache ready")

def fuzzy_match_cache(query: str):
    q = query.lower()
    for key, response in RESPONSE_CACHE.items():
        if response and key in q:
            return response, key
    return None, None

def call_llm_raw(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    if LLM_PROVIDER == "openai":
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages, max_tokens=450,
        )
        return resp.choices[0].message.content
    else:
        resp = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages, max_tokens=450,
        )
        return resp.choices[0].message.content

def call_llm(user_message: str, context: str) -> tuple[str, str, bool]:
    cached, _ = fuzzy_match_cache(user_message)
    if cached:
        return cached, LLM_PROVIDER, True
    return call_llm_raw(user_message + "\n\nSatellite data:\n" + context), LLM_PROVIDER, False

# ── Agent endpoint (auth-gated) ───────────────────────────────────────────────
class AgentQuery(BaseModel):
    query: str
    county_data: dict = {}

@app.post("/api/agent")
def agent_query(req: AgentQuery, user: dict = Depends(verify_token)):
    """Multi-domain agent — requires auth. Enforces daily query limit."""
    user_id = user.get("sub", "local-dev")
    remaining = check_and_increment_usage(user_id)

    waited = 0
    while _county_cache is None and waited < 60:
        time.sleep(2)
        waited += 2

    intent = classify_intent(req.query)

    if intent == "out_of_scope":
        return {
            "response": (
                "I'm a satellite climate intelligence system specialized in: "
                "**kelp/eelgrass restoration** in Puget Sound, **forest fire risk** in the US and India, "
                "and **monsoon/drought risk** for Indian farmers. "
                "Could you ask me something in one of those areas?"
            ),
            "provider": "filter",
            "from_cache": False,
            "data_ready": True,
            "intent": "out_of_scope",
            "queries_remaining": remaining,
        }

    if intent == "fire":
        try:
            fire_data = fire_summary()
        except Exception:
            fire_data = {"us": {}, "india": {}, "data_source": "MODIS"}
        context = json.dumps({"fire": fire_data, "data_source": "MODIS MCD64A1"})
    elif intent == "india":
        try:
            india_data = india_climate_summary()
        except Exception:
            india_data = {}
        context = json.dumps({"india_climate": india_data, "data_source": "CHIRPS/MODIS India"})
    else:
        context = json.dumps(req.county_data if req.county_data else {
            "counties": _county_cache or [],
            "data_source": "Real Landsat GEE NDWI"
        })

    response_text, provider, from_cache = call_llm(req.query, context)
    return {
        "response": response_text,
        "provider": provider,
        "from_cache": from_cache,
        "data_ready": _county_cache is not None,
        "intent": intent,
        "queries_remaining": remaining,
    }

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
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

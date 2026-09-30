"""Climate data: ingestion from Earth Engine, quality checks, snapshots and a source registry.

Every dataset goes through the same path:
  fetch live -> run quality checks -> save a snapshot -> record the run in the registry.
If a live fetch fails, the last snapshot is served and the failure is reported as a
quality issue, never hidden and never replaced with zeros.
"""
import datetime, json, pathlib, threading, time

import ee

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

# ── Source registry ───────────────────────────────────────────────────────────
SOURCES = [
    {"id": "landsat", "name": "Landsat 5, 7, 8 and 9 surface reflectance", "provider": "USGS / NASA, via Google Earth Engine",
     "dataset": "LANDSAT/LT05, LE07, LC08, LC09 C02 T1_L2", "resolution": "30 m", "cadence": "every 16 days per satellite",
     "licence": "US public domain (USGS)", "used_for": "Nearshore vegetation signal, Puget Sound"},
    {"id": "gsw", "name": "Global Surface Water occurrence", "provider": "EC Joint Research Centre, via Google Earth Engine",
     "dataset": "JRC/GSW1_4/GlobalSurfaceWater", "resolution": "30 m", "cadence": "static (1984-2021)",
     "licence": "Free of charge (Copernicus Programme)", "used_for": "Permanent-water mask for the kelp signal"},
    {"id": "firms", "name": "FIRMS active fire detections (MODIS)", "provider": "NASA LANCE, via Google Earth Engine",
     "dataset": "FIRMS", "resolution": "1 km", "cadence": "daily", "licence": "NASA open data",
     "used_for": "Wildfire activity, US and India"},
    {"id": "chirps", "name": "CHIRPS daily rainfall", "provider": "UC Santa Barbara Climate Hazards Center, via Google Earth Engine",
     "dataset": "UCSB-CHG/CHIRPS/DAILY", "resolution": "about 5.5 km", "cadence": "daily, published with a lag of weeks",
     "licence": "Free (see Climate Hazards Center terms)", "used_for": "Rainfall versus the 1991-2020 normal, India"},
    {"id": "modis_ndvi", "name": "MODIS vegetation index (NDVI)", "provider": "NASA LP DAAC, via Google Earth Engine",
     "dataset": "MODIS/061/MOD13A2", "resolution": "1 km", "cadence": "16-day composites", "licence": "NASA open data",
     "used_for": "Vegetation health, India"},
    {"id": "documents", "name": "Public kelp and eelgrass documents", "provider": "WA DNR, WDFW, Northwest Straits Commission and partners",
     "dataset": "backend/kb/manifest.json", "resolution": "page-level passages", "cadence": "rebuilt on demand",
     "licence": "Public agency publications", "used_for": "Knowledge base for search_documents"},
]
_status: dict = {s["id"]: {"state": "not run", "last_success": None, "last_error": None, "checks": []} for s in SOURCES}
_lock = threading.Lock()

def record(source_id: str, ok: bool, checks: list, error: str | None = None):
    with _lock:
        st = _status[source_id]
        st["checks"] = checks
        st["last_attempt"] = _now()
        if ok:
            st["last_success"] = st["last_attempt"]
            st["state"] = "ok" if all(c["passed"] for c in checks) else "warning"
            st["last_error"] = None
        else:
            st["state"] = "failed"
            st["last_error"] = error

def registry() -> list:
    return [{**s, "status": _status[s["id"]]} for s in SOURCES]

def _save(name: str, payload: dict):
    (DATA_DIR / f"{name}_snapshot.json").write_text(json.dumps(payload, indent=1))

def _load(name: str) -> dict | None:
    p = DATA_DIR / f"{name}_snapshot.json"
    return json.loads(p.read_text()) if p.exists() else None

# ── Geometry ──────────────────────────────────────────────────────────────────
COUNTY_BOUNDS = {
    "King": [-122.5, 47.3, -121.9, 47.8], "Skagit": [-122.8, 48.2, -122.1, 48.6],
    "Whatcom": [-122.9, 48.6, -122.1, 49.0], "Kitsap": [-122.9, 47.4, -122.4, 47.9],
    "Pierce": [-122.7, 47.0, -122.1, 47.4], "Snohomish": [-122.5, 47.8, -121.9, 48.2],
}
PUGET_BOUNDS = [-123.2, 47.0, -121.9, 49.0]
US_BOUNDS = [-125.0, 24.0, -66.0, 49.0]
INDIA_BOUNDS = [68.0, 6.0, 97.5, 36.0]

# ── Kelp: nearshore vegetation signal ─────────────────────────────────────────
PERIODS = [
    {"label": "1995", "years": (1995, 1997), "sensor": "TM",   "collections": ["LANDSAT/LT05/C02/T1_L2"], "bands": ("SR_B2", "SR_B3", "SR_B4")},
    {"label": "2000", "years": (1999, 2001), "sensor": "ETM+", "collections": ["LANDSAT/LE07/C02/T1_L2"], "bands": ("SR_B2", "SR_B3", "SR_B4")},
    {"label": "2015", "years": (2014, 2016), "sensor": "OLI",  "collections": ["LANDSAT/LC08/C02/T1_L2"], "bands": ("SR_B3", "SR_B4", "SR_B5")},
    {"label": "2023", "years": (2022, 2024), "sensor": "OLI",  "collections": ["LANDSAT/LC09/C02/T1_L2", "LANDSAT/LC08/C02/T1_L2"], "bands": ("SR_B3", "SR_B4", "SR_B5")},
]
NDVI_THRESHOLD = 0.2
MIN_SCENES = 3
MIN_VALID_SHARE = 0.8
DOCUMENTED_TREND = {
    "direction": "decline",
    "statement": "Bull kelp in South and Central Puget Sound has declined by more than 90% over 150 years, and long-term canopy declines are documented in several areas.",
    "sources": ["Puget Sound Kelp Conservation and Recovery Plan (2020)", "Washington DNR kelp and eelgrass plan"],
}
KELP_METHOD = (f"Share of permanent-water pixels (JRC surface water occurrence of 80% or more) whose summer (June to September) "
               f"median NDVI exceeds {NDVI_THRESHOLD}, from cloud-masked, scaled Landsat surface reflectance.")
KELP_CAVEATS = [
    "This is a screening signal, not a validated measure of kelp canopy area.",
    "It includes any vegetation or algae at the surface, and exposed intertidal vegetation at low tide; tides are not controlled.",
    "County areas are rough bounding boxes, not county boundaries.",
    "1995 and 2000 use older sensors (TM, ETM+) than 2015 and 2023 (OLI), so only 2015 to 2023 is a like-for-like comparison.",
]

def _landsat_ndvi(bands):
    g, r, n = bands
    def f(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        sr = img.select([g, r, n]).multiply(0.0000275).add(-0.2)
        return sr.normalizedDifference([n, r]).rename("ndvi").updateMask(clear)
    return f

def _permanent_water():
    return ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(80).selfMask()

def _period_composite(p, region):
    col = ee.ImageCollection(p["collections"][0])
    for c in p["collections"][1:]:
        col = col.merge(ee.ImageCollection(c))
    y0, y1 = p["years"]
    col = col.filterBounds(region).filterDate(f"{y0}-06-01", f"{y1}-09-30").filter(ee.Filter.lt("CLOUD_COVER", 20))
    return col, col.map(_landsat_ndvi(p["bands"])).median().updateMask(_permanent_water())

def compute_kelp() -> dict:
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(b), {"county": k}) for k, b in COUNTY_BOUNDS.items()])
    region = fc.geometry().bounds()
    water = _permanent_water()
    cells = {c: [] for c in COUNTY_BOUNDS}
    for p in PERIODS:
        t = time.time()
        try:
            col, comp = _period_composite(p, region)
            img = ee.Image.cat([water.rename("water"), comp.mask().selfMask().rename("valid"),
                                comp.gt(NDVI_THRESHOLD).selfMask().rename("veg")])
            res = img.reduceRegions(fc, ee.Reducer.count(), scale=30, tileScale=4).getInfo()
            scenes = col.size().getInfo()
            for f in res["features"]:
                pr = f["properties"]
                w, v, g = pr.get("water", 0), pr.get("valid", 0), pr.get("veg", 0)
                checks = [
                    {"check": "enough_scenes", "passed": scenes >= MIN_SCENES, "detail": f"{scenes} scenes under 20% cloud"},
                    {"check": "valid_pixels", "passed": w > 0 and v / w >= MIN_VALID_SHARE, "detail": f"{v:,} of {w:,} water pixels usable"},
                ]
                cells[pr["county"]].append({"period": p["label"], "sensor": p["sensor"], "scenes": scenes,
                    "signal_pct": round(100 * g / v, 2) if v else None, "status": "ok" if all(c["passed"] for c in checks) else "warning",
                    "checks": checks, "seconds": round(time.time() - t, 1)})
        except Exception as e:
            for c in COUNTY_BOUNDS:
                cells[c].append({"period": p["label"], "sensor": p["sensor"], "scenes": None, "signal_pct": None,
                                 "status": "failed", "checks": [{"check": "query_succeeded", "passed": False, "detail": str(e)[:200]}]})
    counties = []
    for name, tl in cells.items():
        by = {c["period"]: c for c in tl}
        def change(a, b):
            va, vb = by.get(a, {}).get("signal_pct"), by.get(b, {}).get("signal_pct")
            if va is None or vb is None or va == 0:
                return None
            return round(100 * (vb - va) / va, 1)
        counties.append({"county": name, "timeline": tl,
                         "change_2015_2023_pct": change("2015", "2023"),
                         "change_1995_2023_pct": change("1995", "2023"),
                         "current_signal_pct": by.get("2023", {}).get("signal_pct")})
    return {"computed_at": _now(), "metric": "nearshore vegetation signal (%)", "method": KELP_METHOD,
            "caveats": KELP_CAVEATS, "documented_trend": DOCUMENTED_TREND, "counties": counties}

def kelp_quality(snapshot: dict) -> tuple[list, list]:
    """Checks across the whole dataset. Returns (checks, quality_issues)."""
    cells = [c for co in snapshot["counties"] for c in co["timeline"]]
    failed = [c for c in cells if c["status"] == "failed"]
    weak = [c for c in cells if c["status"] == "warning"]
    changes = [c["change_2015_2023_pct"] for c in snapshot["counties"] if c["change_2015_2023_pct"] is not None]
    rising = sum(1 for x in changes if x > 0)
    agrees = not changes or rising <= len(changes) / 2
    checks = [
        {"check": "all_queries_succeeded", "passed": not failed, "detail": f"{len(cells) - len(failed)} of {len(cells)} county-periods computed"},
        {"check": "quality_thresholds_met", "passed": not weak, "detail": f"{len(weak)} county-periods below scene or pixel thresholds"},
        {"check": "sensor_comparability", "passed": False, "detail": "1995 and 2000 (TM, ETM+) are not directly comparable with 2015 and 2023 (OLI); use the 2015-2023 change"},
        {"check": "consistent_with_documented_trend", "passed": agrees,
         "detail": (f"Signal rose in {rising} of {len(changes)} counties from 2015 to 2023, while documents report decline; "
                    "the signal is not a reliable kelp trend" if not agrees else
                    f"Signal fell in {len(changes) - rising} of {len(changes)} counties from 2015 to 2023, consistent with documented decline")},
    ]
    issues = [c["detail"] for c in checks if not c["passed"]]
    return checks, issues

_kelp_cache: dict | None = _load("kelp")

def refresh_kelp(gee_ok: bool, max_age_days: int = 7):
    global _kelp_cache
    if not gee_ok:
        record("landsat", False, [], "Earth Engine not available; serving snapshot")
        return
    if _kelp_cache:
        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(_kelp_cache["computed_at"])
        if age.days < max_age_days:     # recent enough: record the snapshot's checks, skip a 4-minute recompute
            record("landsat", True, kelp_quality(_kelp_cache)[0])
            record("gsw", True, [{"check": "mask_loaded", "passed": True, "detail": "permanent-water mask applied"}])
            return
    try:
        snap = compute_kelp()
        _save("kelp", snap)
        _kelp_cache = snap
        checks, _ = kelp_quality(snap)
        record("landsat", True, checks)
        record("gsw", True, [{"check": "mask_loaded", "passed": True, "detail": "permanent-water mask applied"}])
    except Exception as e:
        record("landsat", False, [], str(e)[:200])

def kelp(county: str | None = None) -> dict:
    if _kelp_cache is None:
        return {"error": "Kelp data has not been computed yet", "quality_issues": ["kelp data unavailable"]}
    snap = _kelp_cache
    checks, issues = kelp_quality(snap)
    live = _status["landsat"]["last_success"] is not None
    if not live:
        issues = [f"live refresh unavailable; snapshot computed {snap['computed_at']}"] + issues
    counties = snap["counties"]
    if county:
        counties = [c for c in counties if c["county"].lower() == county.lower()] or counties
    return {**{k: snap[k] for k in ("computed_at", "metric", "method", "caveats", "documented_trend")},
            "counties": counties, "checks": checks, "quality_issues": issues,
            "status": "live" if live else "snapshot"}

# ── Wildfire ──────────────────────────────────────────────────────────────────
def compute_fire() -> dict:
    firms = ee.ImageCollection("FIRMS")
    last = ee.Date(firms.aggregate_max("system:time_start"))
    end = last.format("YYYY-MM-dd").getInfo()
    window = firms.filterDate(last.advance(-6, "day"), last.advance(1, "day")).select("T21").max().gt(0).selfMask()
    t21 = firms.filterDate(last.advance(-6, "day"), last.advance(1, "day")).select("T21").max()
    out, points = {}, []
    for name, b in (("us", US_BOUNDS), ("india", INDIA_BOUNDS)):
        geom = ee.Geometry.Rectangle(b)
        r = window.reduceRegion(ee.Reducer.count(), geom, 1000, maxPixels=1e10, tileScale=4).getInfo()
        vec = t21.gt(0).selfMask().int().rename("fire").addBands(t21.rename("t21")).reduceToVectors(
            geometry=geom, scale=1000, geometryType="centroid", eightConnected=True,
            reducer=ee.Reducer.max(), maxPixels=1e10, bestEffort=True).limit(4000, "max", False).getInfo()
        clusters = [[round(f["geometry"]["coordinates"][1], 3), round(f["geometry"]["coordinates"][0], 3),
                     round(f["properties"]["max"], 1), name] for f in vec["features"]]
        points += clusters
        out[name] = {"fire_pixels_1km_7d": r.get("T21", 0), "fire_clusters_7d": len(clusters)}
    start = (datetime.date.fromisoformat(end) - datetime.timedelta(days=6)).isoformat()
    return {"computed_at": _now(), "window": f"{start} to {end}", "data_through": end, **out, "points": points,
            "method": "Count of 1 km pixels with at least one active-fire detection (FIRMS, MODIS) in the 7-day window; "
                      "adjacent fire pixels are grouped into clusters, each mapped at its centre with its peak brightness.",
            "caveats": ["A detection is a 1 km pixel where the satellite saw active fire; it is not a count of separate fires.",
                        "Clouds and smoke can hide fires, so counts are a lower bound."],
            "context": {"us_high_risk_states": ["California", "Oregon", "Washington", "Colorado", "Idaho"],
                        "india_high_risk_regions": ["Uttarakhand", "Himachal Pradesh", "Odisha", "Chhattisgarh"],
                        "note": "General context, not data"}}

# ── Drought (India) ───────────────────────────────────────────────────────────
def compute_drought() -> dict:
    india = ee.Geometry.Rectangle(INDIA_BOUNDS)
    ch = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    last = ee.Date(ch.aggregate_max("system:time_start"))
    end = datetime.date.fromisoformat(last.format("YYYY-MM-dd").getInfo())
    start = end - datetime.timedelta(days=29)
    cur = ch.filterDate(start.isoformat(), (end + datetime.timedelta(days=1)).isoformat()).sum()
    d0, d1 = start.timetuple().tm_yday, end.timetuple().tm_yday
    norm = ch.filterDate("1991-01-01", "2021-01-01").filter(ee.Filter.dayOfYear(d0, d1)).sum().divide(30)
    r = ee.Image.cat([cur.rename("cur"), norm.rename("norm")]).reduceRegion(
        ee.Reducer.mean(), india, 5566, maxPixels=1e10, tileScale=4).getInfo()
    nd = ee.ImageCollection("MODIS/061/MOD13A2")
    nlast = ee.Date(nd.aggregate_max("system:time_start"))
    ndvi_date = nlast.format("YYYY-MM-dd").getInfo()
    ndvi = nd.filterDate(nlast, nlast.advance(1, "day")).first().select("NDVI").multiply(0.0001).reduceRegion(
        ee.Reducer.mean(), india, 5000, maxPixels=1e10, tileScale=4).getInfo().get("NDVI")
    rain, normal = round(r["cur"], 1), round(r["norm"], 1)
    pct = round(100 * rain / normal) if normal else None
    category = (None if pct is None else "severe deficit" if pct < 50 else "deficit" if pct < 75
                else "near normal" if pct <= 125 else "above normal")
    return {"computed_at": _now(), "rainfall_window": f"{start.isoformat()} to {end.isoformat()}",
            "rainfall_mm_30d": rain, "normal_mm_30d_1991_2020": normal, "percent_of_normal": pct,
            "rainfall_category": category, "ndvi_mean": round(ndvi, 3) if ndvi is not None else None, "ndvi_date": ndvi_date,
            "thresholds": "percent of normal: under 50 severe deficit, 50-74 deficit, 75-125 near normal, over 125 above normal",
            "method": "Mean over India's bounding box of CHIRPS 30-day rainfall, compared with the 1991-2020 mean for the same days of the year; MODIS NDVI from the latest 16-day composite.",
            "caveats": ["A country-wide average hides regional drought; this is a national screening figure.",
                        "The bounding box includes some neighbouring countries and ocean.",
                        "CHIRPS is published with a lag of several weeks."]}

_cache: dict = {"fire": (_load("fire"), 0.0), "drought": (_load("drought"), 0.0)}
TTL = 3600

def _cached(name: str, source_ids: list, compute, gee_ok: bool, checks_fn) -> dict:
    data, fetched = _cache[name]
    if gee_ok and time.time() - fetched > TTL:
        try:
            data = compute()
            _save(name, data)
            _cache[name] = (data, time.time())
            checks = checks_fn(data)
            for sid in source_ids:
                record(sid, True, checks)
        except Exception as e:
            for sid in source_ids:
                record(sid, False, [], str(e)[:200])
    if data is None:
        return {"error": f"{name} data unavailable", "quality_issues": [f"{name} data unavailable"]}
    checks = checks_fn(data)
    issues = [c["detail"] for c in checks if not c["passed"]]
    if _cache[name][1] == 0.0:
        issues.insert(0, f"live refresh unavailable; snapshot computed {data['computed_at']}")
    return {**data, "checks": checks, "quality_issues": issues, "status": "live" if _cache[name][1] else "snapshot"}

def _fire_checks(d):
    lag = (datetime.date.today() - datetime.date.fromisoformat(d["data_through"])).days
    return [{"check": "fresh", "passed": lag <= 3, "detail": f"data through {d['data_through']} ({lag} days ago)"}]

def _drought_checks(d):
    end = d["rainfall_window"].split(" to ")[1]
    lag = (datetime.date.today() - datetime.date.fromisoformat(end)).days
    return [{"check": "fresh", "passed": lag <= 21, "detail": f"rainfall data ends {end} ({lag} days ago)"},
            {"check": "normal_available", "passed": d["normal_mm_30d_1991_2020"] not in (None, 0), "detail": "1991-2020 normal computed"}]

def fire(gee_ok: bool) -> dict:
    return _cached("fire", ["firms"], compute_fire, gee_ok, _fire_checks)

def _ndvi_checks(d):
    lag = (datetime.date.today() - datetime.date.fromisoformat(d["ndvi_date"])).days
    return [{"check": "fresh", "passed": lag <= 32, "detail": f"latest 16-day composite starts {d['ndvi_date']} ({lag} days ago)"},
            {"check": "value_in_range", "passed": d["ndvi_mean"] is not None and -1 <= d["ndvi_mean"] <= 1, "detail": f"mean NDVI {d['ndvi_mean']}"}]

def drought(gee_ok: bool) -> dict:
    fetched_before = _cache["drought"][1]
    out = _cached("drought", ["chirps"], compute_drought, gee_ok, _drought_checks)
    if out.get("error"):
        return out
    ndvi = _ndvi_checks(out)
    if _cache["drought"][1] != fetched_before:          # a live fetch just happened
        record("modis_ndvi", True, ndvi)
    out["quality_issues"] += [c["detail"] for c in ndvi if not c["passed"]]
    return out

# ── Map tiles ─────────────────────────────────────────────────────────────────
KELP_PALETTE = ["#0b2a4a", "#1d5f7a", "#3a9d8f", "#9bd18b", "#f2e394"]

def kelp_tile(period_label: str) -> str:
    p = next(p for p in PERIODS if p["label"] == period_label)
    _, comp = _period_composite(p, ee.Geometry.Rectangle(PUGET_BOUNDS))
    return comp.getMapId({"min": -0.3, "max": 0.4, "palette": KELP_PALETTE})["tile_fetcher"].url_format

def kelp_change_tile() -> str:
    region = ee.Geometry.Rectangle(PUGET_BOUNDS)
    a = _period_composite(PERIODS[2], region)[1]
    b = _period_composite(PERIODS[3], region)[1]
    return b.subtract(a).getMapId({"min": -0.2, "max": 0.2, "palette": ["#b2182b", "#f7f7f7", "#2166ac"]})["tile_fetcher"].url_format

def drought_tile() -> str:
    ch = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    last = ee.Date(ch.aggregate_max("system:time_start"))
    end = datetime.date.fromisoformat(last.format("YYYY-MM-dd").getInfo())
    start = end - datetime.timedelta(days=29)
    cur = ch.filterDate(start.isoformat(), (end + datetime.timedelta(days=1)).isoformat()).sum()
    norm = ch.filterDate("1991-01-01", "2021-01-01").filter(
        ee.Filter.dayOfYear(start.timetuple().tm_yday, end.timetuple().tm_yday)).sum().divide(30)
    pct = cur.divide(norm.max(1)).multiply(100).clip(ee.Geometry.Rectangle(INDIA_BOUNDS))
    return pct.getMapId({"min": 25, "max": 175, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]})["tile_fetcher"].url_format

def fire_tile() -> str:
    firms = ee.ImageCollection("FIRMS")
    last = ee.Date(firms.aggregate_max("system:time_start"))
    img = firms.filterDate(last.advance(-6, "day"), last.advance(1, "day")).select("T21").max()
    return img.getMapId({"min": 300, "max": 400, "palette": ["#ffd166", "#f77f00", "#d62828"]})["tile_fetcher"].url_format

# ── Illustrative restoration sites ────────────────────────────────────────────
ILLUSTRATIVE_SITES = [
    {"id": "S01", "name": "Nisqually Delta eelgrass", "county": "Pierce", "lat": 47.08, "lng": -122.70, "acres": 180, "cost_usd": 900000, "salmon_benefit": "High"},
    {"id": "S02", "name": "Skagit Bay eelgrass", "county": "Skagit", "lat": 48.33, "lng": -122.47, "acres": 240, "cost_usd": 1200000, "salmon_benefit": "High"},
    {"id": "S03", "name": "Port Susan nearshore", "county": "Snohomish", "lat": 48.15, "lng": -122.40, "acres": 95, "cost_usd": 380000, "salmon_benefit": "High"},
    {"id": "S04", "name": "Hood Canal kelp", "county": "Kitsap", "lat": 47.62, "lng": -122.95, "acres": 320, "cost_usd": 2400000, "salmon_benefit": "High"},
    {"id": "S05", "name": "Padilla Bay eelgrass", "county": "Skagit", "lat": 48.52, "lng": -122.52, "acres": 410, "cost_usd": 1640000, "salmon_benefit": "Medium"},
    {"id": "S06", "name": "Commencement Bay nearshore", "county": "Pierce", "lat": 47.27, "lng": -122.44, "acres": 75, "cost_usd": 525000, "salmon_benefit": "High"},
    {"id": "S07", "name": "Possession Sound kelp", "county": "Snohomish", "lat": 47.95, "lng": -122.30, "acres": 130, "cost_usd": 780000, "salmon_benefit": "High"},
    {"id": "S08", "name": "Duckabush estuary", "county": "Kitsap", "lat": 47.65, "lng": -122.93, "acres": 285, "cost_usd": 1710000, "salmon_benefit": "High"},
    {"id": "S09", "name": "Drayton Harbor eelgrass", "county": "Whatcom", "lat": 48.98, "lng": -122.76, "acres": 88, "cost_usd": 264000, "salmon_benefit": "Medium"},
]
BENEFIT_WEIGHT = {"High": 1.5, "Medium": 1.0, "Low": 0.5}

def rank_sites(budget_usd: float | None = None) -> dict:
    ranked = []
    for s in ILLUSTRATIVE_SITES:
        cost_per_acre = s["cost_usd"] / s["acres"]
        score = round(BENEFIT_WEIGHT[s["salmon_benefit"]] * 1000 / (cost_per_acre / 1000), 1)
        ranked.append({**s, "cost_per_acre_usd": round(cost_per_acre), "score": score})
    ranked.sort(key=lambda s: s["score"], reverse=True)
    funded, left = [], budget_usd
    if budget_usd:
        for s in ranked:
            if s["cost_usd"] <= left:
                funded.append(s["id"])
                left -= s["cost_usd"]
    return {"illustrative": True,
            "notice": "Illustrative sites: real place names, but acreage and costs are examples, not official ESRP records.",
            "scoring": "score = salmon-benefit weight (High 1.5, Medium 1.0, Low 0.5) x 1000 / cost per acre in $1,000s. "
                       "The satellite signal is not used because it failed validation against documented trends.",
            "budget_usd": budget_usd, "funded_within_budget": funded,
            "budget_remaining_usd": round(left) if budget_usd else None, "sites": ranked}


# ── Kelp: yearly series ───────────────────────────────────────────────────────
YEARS = list(range(1990, 2026))

def _year_sensor(y: int) -> dict:
    if y <= 2011:
        return {"sensor": "TM", "collections": ["LANDSAT/LT05/C02/T1_L2"], "bands": ("SR_B2", "SR_B3", "SR_B4")}
    if y == 2012:
        return {"sensor": "ETM+", "collections": ["LANDSAT/LE07/C02/T1_L2"], "bands": ("SR_B2", "SR_B3", "SR_B4")}
    cols = ["LANDSAT/LC08/C02/T1_L2"] + (["LANDSAT/LC09/C02/T1_L2"] if y >= 2022 else [])
    return {"sensor": "OLI", "collections": cols, "bands": ("SR_B3", "SR_B4", "SR_B5")}

def compute_kelp_yearly(years=YEARS) -> dict:
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(b), {"county": k}) for k, b in COUNTY_BOUNDS.items()])
    region = fc.geometry().bounds()
    series = {c: [] for c in COUNTY_BOUNDS}
    for y in years:
        p = {**_year_sensor(y), "label": str(y), "years": (y, y)}
        try:
            col, comp = _period_composite(p, region)
            img = ee.Image.cat([comp.rename("ndvi_mean"), comp.gt(NDVI_THRESHOLD).rename("veg")])
            res = img.reduceRegions(fc, ee.Reducer.mean(), scale=30, tileScale=4).getInfo()
            scenes = col.size().getInfo()
            for f in res["features"]:
                pr = f["properties"]
                ok = scenes >= MIN_SCENES and pr.get("ndvi_mean") is not None
                series[pr["county"]].append({
                    "year": y, "sensor": p["sensor"], "scenes": scenes,
                    "ndvi_mean": round(pr["ndvi_mean"], 4) if pr.get("ndvi_mean") is not None else None,
                    "signal_pct": round(100 * pr["veg"], 2) if pr.get("veg") is not None else None,
                    "status": "ok" if ok else "warning"})
        except Exception as e:
            for c in COUNTY_BOUNDS:
                series[c].append({"year": y, "sensor": p["sensor"], "scenes": None, "ndvi_mean": None,
                                  "signal_pct": None, "status": "failed", "error": str(e)[:160]})
    return {"computed_at": _now(), "years": years, "method": "Summer (June to September) median NDVI over permanent water, per year; "
            "signal = share of those pixels above NDVI 0.2.", "counties": series}

def compute_sst_yearly(years=YEARS) -> dict:
    """Summer sea-surface temperature for the Salish Sea box, with anomalies against 1991-2020."""
    oi = ee.ImageCollection("NOAA/CDR/OISST/V2_1").select("sst")
    box = ee.Geometry.Rectangle(SALISH_BOUNDS)
    def summer(y):
        y = ee.Number(y)
        img = oi.filter(ee.Filter.calendarRange(y, y, "year")).filter(ee.Filter.calendarRange(6, 9, "month")).mean().multiply(0.01)
        return ee.Feature(None, img.reduceRegion(ee.Reducer.mean(), box, 27830)).set("year", y)
    feats = ee.FeatureCollection(ee.List.sequence(1991, max(years)).map(summer)).getInfo()["features"]
    vals = {int(f["properties"]["year"]): f["properties"].get("sst") for f in feats}
    base = [vals[y] for y in range(1991, 2021) if vals.get(y) is not None]
    normal = sum(base) / len(base)
    return {"computed_at": _now(), "region": "Salish Sea and Strait of Juan de Fuca box " + str(SALISH_BOUNDS),
            "normal_1991_2020_c": round(normal, 2),
            "years": [{"year": y, "summer_sst_c": round(v, 2), "anomaly_c": round(v - normal, 2)}
                      for y, v in sorted(vals.items()) if v is not None],
            "method": "NOAA OISST v2.1 daily sea-surface temperature, June to September mean over the box, "
                      "anomaly against the 1991-2020 summer mean.",
            "caveats": ["OISST is 0.25 degree (about 25 km), so inland Puget Sound is barely resolved; this tracks the wider Salish Sea and coast."]}

SALISH_BOUNDS = [-125.0, 47.0, -122.0, 49.0]

def _yearly_quality(k: dict, sst: dict | None) -> list:
    cells = [c for s in k["counties"].values() for c in s]
    failed = [c for c in cells if c["status"] == "failed"]
    weak = [c for c in cells if c["status"] == "warning"]
    checks = [
        {"check": "all_years_computed", "passed": not failed, "detail": f"{len(cells) - len(failed)} of {len(cells)} county-years computed"},
        {"check": "enough_scenes_each_year", "passed": not weak,
         "detail": f"{len(weak)} county-years had fewer than {MIN_SCENES} clear scenes" if weak else "every year has enough scenes"},
        {"check": "single_sensor_series", "passed": False,
         "detail": "Sensor changes in 2012 (TM to ETM+) and 2013 (to OLI); compare years within one sensor era"},
    ]
    # Year-to-year noise within one sensor era: if single years swing wildly, single years can't carry a trend.
    swings = []
    for rows in k["counties"].values():
        vals = [r["signal_pct"] for r in rows if r["sensor"] == "OLI" and r["signal_pct"]]
        swings += [abs(b - a) / a for a, b in zip(vals, vals[1:]) if a]
    if swings:
        med = sorted(swings)[len(swings) // 2]
        checks.append({"check": "stable_between_years", "passed": med <= 0.5,
                       "detail": f"median year-to-year change in the signal is {med:.0%} within 2013-2025; "
                                 + ("single years are too noisy to read as a trend; use multi-year periods" if med > 0.5 else "acceptable")})
    if sst:
        checks.append({"check": "sst_available", "passed": len(sst["years"]) >= 30, "detail": f"{len(sst['years'])} summers of sea-surface temperature"})
    return checks

_yearly_cache = _load("kelp_yearly")
_sst_cache = _load("sst")

def refresh_yearly(gee_ok: bool, max_age_days: int = 30):
    """Yearly series change slowly: recompute only when the snapshot is older than max_age_days."""
    global _yearly_cache, _sst_cache
    if not gee_ok:
        return
    def stale(snap):
        if not snap:
            return True
        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(snap["computed_at"])
        return age.days >= max_age_days
    try:
        if stale(_sst_cache):
            _sst_cache = compute_sst_yearly(); _save("sst", _sst_cache)
        if stale(_yearly_cache):
            _yearly_cache = compute_kelp_yearly(); _save("kelp_yearly", _yearly_cache)
    except Exception as e:
        record("landsat", False, [], f"yearly series: {str(e)[:160]}")

def kelp_yearly(county: str | None = None) -> dict:
    if not _yearly_cache:
        return {"error": "Yearly kelp series has not been computed yet", "quality_issues": ["yearly kelp series unavailable"]}
    k = _yearly_cache
    checks = _yearly_quality(k, _sst_cache)
    counties = k["counties"]
    if county:
        counties = {c: v for c, v in counties.items() if c.lower() == county.lower()} or counties
    return {"computed_at": k["computed_at"], "method": k["method"], "counties": counties, "sst": _sst_cache,
            "checks": checks, "quality_issues": [c["detail"] for c in checks if not c["passed"]],
            "caveats": KELP_CAVEATS}

# ── El Niño (ENSO) ────────────────────────────────────────────────────────────
NINO34 = [-170.0, -5.0, -120.0, 5.0]

def compute_enso() -> dict:
    import requests
    txt = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=20).text
    rows = []
    for line in txt.strip().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 4:
            rows.append({"season": parts[0], "year": int(parts[1]), "total_c": float(parts[2]), "oni": float(parts[3])})
    latest = rows[-1]
    def phase(v):
        return "El Niño" if v >= 0.5 else "La Niña" if v <= -0.5 else "Neutral"
    def strength(v):
        a = abs(v)
        return None if a < 0.5 else "weak" if a < 1.0 else "moderate" if a < 1.5 else "strong" if a < 2.0 else "very strong"
    oi = ee.ImageCollection("NOAA/CDR/OISST/V2_1")
    last = ee.Date(oi.aggregate_max("system:time_start"))
    oisst_date = last.format("YYYY-MM-dd").getInfo()
    anom = oi.filterDate(last.advance(-29, "day"), last.advance(1, "day")).select("anom").mean().multiply(0.01) \
             .reduceRegion(ee.Reducer.mean(), ee.Geometry.Rectangle(NINO34), 27830).getInfo().get("anom")
    return {"computed_at": _now(),
            "oni_latest": {**latest, "phase": phase(latest["oni"]), "strength": strength(latest["oni"])},
            "oni_series": rows[-60:],
            "nino34_oisst_30d": {"anomaly_c": round(anom, 2) if anom is not None else None, "through": oisst_date,
                                 "baseline": "OISST climatology (1971-2000)"},
            "method": "Official ENSO status from the NOAA CPC Oceanic Niño Index (3-month running Niño 3.4 anomaly, centred "
                      "30-year base periods). Cross-checked with the latest 30-day Niño 3.4 anomaly from NOAA OISST.",
            "thresholds": "ONI of +0.5 or more for five overlapping seasons is El Niño; -0.5 or less is La Niña. "
                          "Strength: 0.5-0.9 weak, 1.0-1.4 moderate, 1.5-1.9 strong, 2.0+ very strong.",
            "caveats": ["The ONI is a 3-month average, so it lags current conditions by about a month.",
                        "OISST anomalies use an older, cooler baseline, so they run higher than the ONI; compare direction, not size."],
            "pacific_northwest_note": "El Niño winters in the Pacific Northwest tend to be warmer and drier than normal."}

def _enso_checks(d):
    o, s = d["oni_latest"], d["nino34_oisst_30d"]
    a = s["anomaly_c"]
    if a is None:
        agree = True
    elif abs(o["oni"]) < 0.5:
        agree = abs(a) < 1.5            # neutral ONI: OISST should not show a strong event
    else:
        agree = (o["oni"] > 0) == (a > 0)
    lag = (datetime.date.today() - datetime.date.fromisoformat(s["through"])).days if s["through"] else 99
    return [{"check": "oni_recent", "passed": o["year"] >= datetime.date.today().year - (1 if datetime.date.today().month <= 2 else 0),
             "detail": f"latest ONI season {o['season']} {o['year']}"},
            {"check": "oisst_fresh", "passed": lag <= 7, "detail": f"OISST through {s['through']} ({lag} days ago)"},
            {"check": "sources_agree_on_phase", "passed": agree,
             "detail": f"ONI {o['oni']:+.2f} and OISST Niño 3.4 {s['anomaly_c']:+.2f} °C point the same way" if agree
                       else f"ONI {o['oni']:+.2f} and OISST {s['anomaly_c']:+.2f} °C disagree on phase"}]

_cache["enso"] = (_load("enso"), 0.0)
SOURCES.append({"id": "enso", "name": "El Niño: Oceanic Niño Index and OISST", "provider": "NOAA Climate Prediction Center; NOAA OISST via Google Earth Engine",
                "dataset": "CPC oni.ascii.txt; NOAA/CDR/OISST/V2_1", "resolution": "Niño 3.4 region average; 0.25 degree",
                "cadence": "ONI monthly; OISST daily", "licence": "US government public domain (NOAA)",
                "used_for": "El Niño status, and Salish Sea summer sea temperature for kelp"})
_status["enso"] = {"state": "not run", "last_success": None, "last_error": None, "checks": []}

def enso(gee_ok: bool) -> dict:
    return _cached("enso", ["enso"], compute_enso, gee_ok, _enso_checks)

def sst_anomaly_tile() -> str:
    oi = ee.ImageCollection("NOAA/CDR/OISST/V2_1")
    last = ee.Date(oi.aggregate_max("system:time_start"))
    img = oi.filterDate(last.advance(-6, "day"), last.advance(1, "day")).select("anom").mean().multiply(0.01)
    return img.getMapId({"min": -3, "max": 3, "palette": ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"]})["tile_fetcher"].url_format


# ── Fallback map tiles (NASA GIBS, no key) ────────────────────────────────────
def gibs_tile(layer: str, matrix: str, days_ago: int = 2) -> str:
    d = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
    return f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{d}/{matrix}/{{z}}/{{y}}/{{x}}.png"

FALLBACK_TILES = {
    "sst": {"tile_url": gibs_tile("GHRSST_L4_MUR_Sea_Surface_Temperature_Anomalies", "GoogleMapsCompatible_Level7"),
            "fallback": True, "note": "Fallback layer: NASA GIBS MUR sea-surface temperature anomaly (daily), not the OISST 7-day mean."},
    "drought": {"tile_url": gibs_tile("IMERG_Precipitation_Rate", "GoogleMapsCompatible_Level6"),
                "fallback": True, "note": "Fallback layer: NASA GIBS IMERG daily precipitation rate, not rainfall as a percent of normal."},
}

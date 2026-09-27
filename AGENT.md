# KelpWatch — Agentic AI Architecture & Auditability

## What "agentic" means here

KelpWatch's AI layer is not a general-purpose chatbot. It is a **domain-constrained, satellite-grounded agent** that:

1. Classifies every incoming query by domain before touching an LLM
2. Fetches live satellite data for that domain from Google Earth Engine
3. Injects that data as grounding context into the LLM prompt
4. Returns a response that cites specific numbers the satellite data produced
5. Records the query against the authenticated user's daily quota in Supabase

No query reaches the LLM without passing through steps 1–2. No response is returned without the satellite numbers in step 3 being present in the prompt the LLM received.

---

## System diagram

```
User query (POST /api/agent)
        │
        ▼
┌───────────────────┐
│  JWT verification │  ← Supabase JWT, HS256
│  (verify_token)   │
└────────┬──────────┘
         │ user_id
         ▼
┌────────────────────┐
│  Daily quota check │  ← Supabase token_usage table
│  (check_and_inc)   │    raises HTTP 429 when exceeded
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Intent classifier │  ← keyword matching, no LLM involved
│  (classify_intent) │    returns: kelp | fire | india | out_of_scope
└────────┬───────────┘
         │
    ┌────┴────────────────────┐
    │                         │
  out_of_scope          kelp / fire / india
    │                         │
    ▼                         ▼
  Static refusal      Live GEE satellite fetch
  (no LLM call)       (fire_summary / india_climate_summary
                        / cached county NDWI)
                             │
                             ▼
                    ┌─────────────────┐
                    │  Response cache │  ← checked first; 20 pre-warmed keys
                    │  (fuzzy_match)  │    exact substring match on query
                    └────────┬────────┘
                             │ miss
                             ▼
                    ┌─────────────────┐
                    │  LLM call       │  ← system prompt + satellite JSON
                    │  Groq / OpenAI  │
                    └────────┬────────┘
                             │
                             ▼
                    Response + metadata
                    (intent, provider, from_cache,
                     queries_remaining, data_ready)
```

---

## Intent routing

`classify_intent` (backend/main.py:477) runs **before** any LLM call and requires **zero tokens**.

```
query → lowercase → keyword scan across three domain lists → domain with most hits
```

**Domain keyword sets** (abbreviated):

| Domain | Sample keywords |
|---|---|
| `kelp` | kelp, eelgrass, ndwi, puget, wdfw, esrp, salmon, orca, restoration, 2040 |
| `fire` | fire, wildfire, forest fire, burn, modis, uttarakhand, fire risk, fire season |
| `india` | india, monsoon, rainfall, drought, farmer, crop, kharif, chirps, precipitation |

**Out-of-scope signals** short-circuit all domain matching:

```
recipe, cook, bake, restaurant, movie, song, write code, programming,
javascript, python code, sql query, joke, poem, story, translate,
stock price, crypto, bitcoin, medical advice, diagnosis, legal advice, tax advice
```

When `out_of_scope` fires, the response is a static string — no GEE call, no LLM call, no quota consumed.

Default when ambiguous (zero hits on all domains): routes to `kelp`.

---

## Satellite data grounding

Every LLM prompt contains a `Satellite data:` section with live JSON pulled from GEE seconds before the LLM call. The LLM cannot fabricate numbers it was not given.

### Kelp domain

Source: **Landsat 8/9 C02 + Landsat TM5 C02** via GEE  
Metric: NDWI (Normalized Difference Water Index) = `(Green − NIR) / (Green + NIR)`  
Coverage: 6 Puget Sound counties × 4 time periods (1995–97, 1999–2001, 2009–11, 2022–24)  
Scale: 30 m spatial resolution, cloud cover < 20% filter applied

Context injected into prompt:
```json
{
  "counties": [
    { "county": "King", "degradation_pct": -18.4, "current_ndwi": -0.193,
      "timeline": [{"year":"1995","ndwi":-0.165}, ...] }
  ],
  "goal": { "target_acres": 10000, "restored_acres": 1847, "trajectory_year": 2051 },
  "data_source": "Real Landsat GEE satellite NDWI"
}
```

### Fire domain

Source: **MODIS MCD64A1 Burned Area** via GEE  
Metric: Burned pixel count at 5 km scale; estimated km² = pixels × 25  
Coverage: Continental US + Indian subcontinent, last 30 days rolling

Context injected:
```json
{
  "fire": {
    "us":    { "burned_pixels_5km": 412, "estimated_km2": 10300 },
    "india": { "burned_pixels_5km": 88,  "estimated_km2": 2200 }
  },
  "data_source": "MODIS MCD64A1 Burned Area · 500m · NASA"
}
```

### India domain

Sources:
- **CHIRPS Daily** (UCSB-CHG) — rainfall mm accumulated last 30 days, ~5 km resolution
- **MODIS MOD13A2** — NDVI (vegetation health), 1 km resolution, last 30 days

Drought heuristics applied server-side before LLM sees the data:
- `rainfall_mm < 50` → `drought_risk = High`
- `ndvi < 3000` (scaled ×10000) → `crop_stress = Severe`

Context injected:
```json
{
  "india_climate": {
    "rainfall_mm_30d": 42.3,
    "ndvi_mean": 2840,
    "drought_risk": "High",
    "crop_stress": "Severe",
    "farmer_advisory": "Drought alert — below-normal rainfall. Kharif sowing at risk."
  }
}
```

---

## System prompt constraints

The LLM receives this instruction set on every call (backend/main.py:490):

```
RESPONSE RULES:
1. Cite specific satellite numbers — NDWI, rainfall mm, burn area km², NDVI values
2. Connect data to real-world impact (farmers, ecosystems, communities, policy)
3. Give actionable recommendations — dollar amounts, percentages, season timelines
4. Use bullet points. Under 260 words. Direct language.
5. For out-of-domain questions, say so briefly and redirect to supported topics.
```

This means auditors can verify any response by checking whether the numbers cited appear in the satellite context JSON that was injected.

---

## Response cache

On startup, after GEE county data is ready, the backend pre-warms a 20-key in-memory cache by calling the LLM with the most common queries (backend/main.py:560). Cache lookup is a simple substring match on the incoming query.

**Cache hit → zero LLM tokens consumed, instant response.**

The response object always includes `"from_cache": true/false` so the caller knows whether the answer came from a live LLM call or a pre-warmed response.

Cache keys span all three domains:
- Kelp: `esrp`, `salmon`, `orca`, `2040 goal`, `king county`, `skagit`, `roi`, `nearshore`, `duckabush`, ...
- Fire: `wildfire`, `india fire`, `fire risk`
- India: `monsoon`, `india drought`, `farmer risk`, `kharif`

---

## Auditability fields in every response

```json
{
  "response":          "...",
  "intent":            "kelp | fire | india | out_of_scope",
  "provider":          "groq | openai | filter",
  "from_cache":        true,
  "data_ready":        true,
  "queries_remaining": 4
}
```

| Field | What it proves |
|---|---|
| `intent` | Which domain the classifier chose — reviewable against keyword lists |
| `provider` | Which model produced the text (or `filter` = no LLM was called) |
| `from_cache` | Whether the response was pre-generated at startup vs live |
| `data_ready` | Whether GEE county data had finished loading when the query ran |
| `queries_remaining` | User's remaining daily quota at time of response |

---

## Auth and usage audit trail

**Authentication**: Supabase JWT (HS256). Every call to `/api/agent` and `/api/usage` must carry a valid `Authorization: Bearer <token>` header. The `sub` claim becomes the `user_id`.

**Usage ledger**: Every successful agent call increments `token_usage(user_id, query_date, query_count)` in Supabase. The table is append-on-new-date, update-on-same-date — giving a per-user, per-day audit trail.

```sql
select user_id, query_date, query_count
from token_usage
order by query_date desc, query_count desc;
```

**Local dev bypass**: When `SUPABASE_JWT_SECRET` is not set, `verify_token` returns `{"sub": "local-dev"}` and `check_and_increment_usage` short-circuits to return the full daily limit without touching Supabase. This bypass is explicit in code and only activates when the env var is absent — it cannot be triggered by a client request.

---

## What the agent cannot do

| Capability | Status |
|---|---|
| Answer questions outside kelp / fire / India domains | Blocked by intent classifier |
| Call external APIs beyond GEE, Groq/OpenAI, Supabase | No — only these three are wired |
| Modify satellite data or Supabase records | No — all GEE calls are read-only; Supabase writes are only to `token_usage` and `contact_requests` |
| Exceed daily query limit per user | Blocked by Supabase `token_usage` check before LLM is called |
| Run without satellite context in the prompt | No — context is always injected; the LLM never receives a bare user query |

---

## Verifying a response

To confirm any AI response is grounded in real data:

1. Hit `/api/counties`, `/api/fire/summary`, or `/api/india/summary` directly — these return the raw satellite numbers with no LLM involved.
2. Compare the numbers cited in the agent's response against the raw endpoint output.
3. Check `"from_cache"` in the response — if `true`, the grounding was the startup-time snapshot; if `false`, it was live GEE data fetched at query time.
4. The `"data_ready"` flag tells you whether the GEE precompute had completed; if `false`, kelp responses used an empty county list.

All satellite queries are reproducible: re-running the same GEE collection filters (date range, cloud cover threshold, geometry) in the Earth Engine Code Editor will produce the same NDWI / rainfall / NDVI values.

# KelpWatch: how the agent works

KelpWatch is a climate data agent for Puget Sound kelp and eelgrass, wildfire, and drought risk in India. It is built to show one idea: an agent is only as good as its data, so every step between the data and the answer is checked.

## The path of every question

```
question
  │
  ▼
1. Input policy (code)        scope rule, personal-data redaction        backend/policy.py
  │   refused? → fixed reply, no model call
  ▼
2. Agent loop (model)         plans, loads a skill, calls approved tools  backend/agent.py
  │   each tool call checked against the allowlist
  │   each tool result capped to fit the context budget
  ▼
3. Verification (code)        every figure in the answer matched         backend/verify.py
  │                           against the tool results
  ▼
4. Output policy (code)       evidence, citations, illustrative-data,    backend/policy.py
  │                           data-quality disclosure, personal data
  ▼
answer + citations + verification + policy results + full trace
```

The model writes the answer. It never decides whether a rule passed: every check in steps 1, 3 and 4 is deterministic code.

## Data ingestion and quality

| Source | Dataset | Used for | Quality checks |
|---|---|---|---|
| Landsat 5, 7, 8, 9 | `LANDSAT/*/C02/T1_L2` via Earth Engine | Nearshore vegetation signal, 6 Puget Sound areas, 4 periods | scenes per period, usable pixels, sensor comparability, consistency with documented trend |
| JRC Global Surface Water | `JRC/GSW1_4/GlobalSurfaceWater` | Permanent-water mask | mask loaded |
| NASA FIRMS | `FIRMS` via Earth Engine | 7-day active fire pixels, US and India | data freshness |
| CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | 30-day rainfall versus the 1991-2020 normal, India | freshness, normal available |
| MODIS NDVI | `MODIS/061/MOD13A2` | Vegetation health, India | freshness |
| Public documents | `backend/kb/manifest.json` | Knowledge base | readable pages, redaction, chunk sizes, retrieval test |

Each dataset follows the same path in `backend/climate.py`: fetch live, run checks, save a snapshot to `backend/data/`, record the run in the source registry. If a live fetch fails, the last snapshot is served and the failure is reported. Failed values are never replaced with zeros.

### The kelp signal, and why it is labelled a screening signal

The measure is the share of permanent-water pixels whose summer median NDVI exceeds 0.2, from cloud-masked, scaled Landsat surface reflectance. Its checks fail on purpose where the science is weak:

- **Sensor comparability** always fails for 1995 and 2000 against 2015 and 2023, because TM and ETM+ differ from OLI. Only the 2015 to 2023 change is like-for-like.
- **Consistency with the documented trend** compares the satellite change with the decline reported by the Puget Sound Kelp Conservation and Recovery Plan and WA DNR. When the signal rises while the documents report decline, the check fails and the agent must say so.

Tides are not controlled, the signal includes any surface vegetation or algae, and the areas are bounding boxes, not county boundaries. The agent reports these caveats with the data.

## Knowledge base

`backend/ingest_docs.py` builds the knowledge base from six public documents (WA DNR, WDFW, Northwest Straits Commission and partners):

1. **Extract** text per page with pypdf; image-only pages are flagged as needing OCR.
2. **Redact** emails and phone numbers before anything is stored.
3. **Chunk** on paragraph and sentence boundaries, never across pages, and check that chunks fall within 200 to 1,500 characters.
4. **Label** every passage with document, publisher, year, page and URL.

Retrieval is BM25 keyword search (`backend/kb.py`) with a small domain thesaurus. A labelled set of 12 questions, including vocabulary-gap questions, is scored at startup, with and without the thesaurus.

## Policy pack

`backend/policy.yaml` holds the rules in plain language, each mapped to a check in code:

| Rule | Stage | If it fails |
|---|---|---|
| scope | input | refuse, no model call |
| personal-data-input | input | redact |
| approved-tools | tool call | block the call |
| evidence (90% of figures traced) | output | flag |
| citations | output | flag |
| illustrative-data | output | add a notice |
| data-quality | output | add a notice |
| personal-data-output | output | redact |

## Skills

`backend/skills/*.md` are step-by-step procedures the agent loads with `use_skill`: restoration prioritisation, wildfire briefing, and drought advisory.

## Tools

`get_kelp_trends`, `get_wildfire_activity`, `get_drought_conditions`, `search_documents`, `rank_restoration_sites` (illustrative sites only), `get_data_quality`, `use_skill`.

## What every answer returns

```json
{
  "answer": "...",
  "citations": [{"ref": "D1", "title": "...", "page": 12, "url": "..."}],
  "verification": {"figures_checked": 6, "figures_verified": 6, "unverified": []},
  "policy": [{"rule": "evidence", "passed": true, "detail": "6 of 6 figures traced to data"}],
  "trace": [{"type": "policy"}, {"type": "model"}, {"type": "tool"}, {"type": "verification"}],
  "tools_used": ["use_skill", "search_documents", "get_kelp_trends"]
}
```

## Transparency endpoints

| Endpoint | Shows |
|---|---|
| `/api/sources` | source registry, run status, checks, and the ingestion report |
| `/api/evals` | retrieval test results |
| `/api/policy` | the policy pack |
| `/api/agent/config` | model, tools and skills |
| `/api/kb/search?q=` | knowledge-base search, the same as the agent's tool |

## Access

Sign-in is through Supabase (ES256 tokens verified against the project's public keys). Each user has a daily question limit recorded in Supabase. Without `SUPABASE_URL`, the server runs open for local development.

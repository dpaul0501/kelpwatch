# KelpWatch — Satellite Climate Intelligence

> Real satellite data. Real AI. Real impact.  
> Kelp restoration · Forest fire risk · India monsoon & farmer risk

---

## Architecture

```
frontend/          Static HTML/JS — no build step, deployed on Vercel
backend/           FastAPI — deployed on Railway
  main.py          All API logic + GEE + LLM agent
  requirements.txt Python dependencies
railway.toml       Railway build + start config
vercel.json        Vercel SPA rewrite config
frontend/config.js API base URL + Supabase keys (update for prod)
```

**API endpoints**

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/status` | none | GEE county cache readiness |
| GET | `/api/tiles/current` | none | Landsat 8/9 NDWI tile (2022–2024) |
| GET | `/api/tiles/historical` | none | Landsat TM5 NDWI tile (1995–1997) |
| GET | `/api/tiles/fires` | none | MODIS burned area tile (last 30d) |
| GET | `/api/tiles/india-rainfall` | none | CHIRPS rainfall tile over India |
| GET | `/api/tiles/india-ndvi` | none | MODIS NDVI tile over India |
| GET | `/api/counties` | none | 6-county Puget Sound NDWI timeline |
| GET | `/api/goal-tracker` | none | WA DNR 2040 restoration progress |
| GET | `/api/esrp-sites` | none | ESRP grant sites ranked by ROI |
| POST | `/api/rank-grants` | none | Custom grant ranking from payload |
| GET | `/api/fire/summary` | none | US + India MODIS fire stats |
| GET | `/api/india/summary` | none | CHIRPS + NDVI farmer risk stats |
| POST | `/api/agent` | JWT | Multi-domain AI agent (usage-limited) |
| GET | `/api/usage` | JWT | Queries used today vs daily limit |
| POST | `/api/contact` | none | Enterprise contact form |

---

## Local development

```bash
cd backend

# Create and activate venv
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# One-time GEE authentication (application-default credentials)
earthengine authenticate

# Create backend/.env
cat > .env <<'EOF'
GEE_PROJECT=kelpwatch-2026
GROQ_API_KEY=sk-...          # free at console.groq.com
# OPENAI_API_KEY=...         # optional — set LLM_PROVIDER=openai to use
# LLM_PROVIDER=groq          # groq (default) or openai
# DAILY_QUERY_LIMIT=5        # default 5
# SUPABASE_URL=...           # omit → local dev bypass (no auth, no usage tracking)
# SUPABASE_SERVICE_KEY=...
# SUPABASE_JWT_SECRET=...
EOF

# Start server on port 8100
uvicorn main:app --reload --port 8100
```

Open `frontend/index.html` in your browser — no build step needed.

Without Supabase env vars the backend bypasses JWT verification and usage limits
so you can test the AI agent locally without signing in.

---

## Deploy to production

### Step 1 — Supabase (auth + usage tracking)

Create a new project at [supabase.com](https://supabase.com), then run in **SQL Editor**:

```sql
-- Usage tracking per user per day
create table token_usage (
  id           uuid    default gen_random_uuid() primary key,
  user_id      uuid    references auth.users(id) on delete cascade not null,
  query_date   date    not null default current_date,
  query_count  integer not null default 0,
  unique(user_id, query_date)
);

-- Enterprise contact form submissions
create table contact_requests (
  id          uuid default gen_random_uuid() primary key,
  name        text not null,
  email       text not null,
  company     text,
  message     text not null,
  created_at  timestamptz default now()
);

-- Row-level security
alter table token_usage enable row level security;
create policy "own usage" on token_usage for select using (auth.uid() = user_id);

alter table contact_requests enable row level security;
create policy "insert contact" on contact_requests for insert with check (true);
```

From **Settings → API** copy:

| Key | Where to use |
|---|---|
| Project URL | `SUPABASE_URL` (backend) + `frontend/config.js` |
| `anon` public key | `supabaseAnonKey` in `frontend/config.js` |
| `service_role` secret key | `SUPABASE_SERVICE_KEY` (backend only — never expose) |
| JWT secret (Settings → API → JWT Settings) | `SUPABASE_JWT_SECRET` (backend) |

Enable **Email** auth provider under **Authentication → Providers**.

---

### Step 2 — Google Earth Engine service account

1. Open [Google Cloud Console](https://console.cloud.google.com) → the project that owns your GEE quota.
2. **IAM & Admin → Service Accounts → Create Service Account**.
3. Grant it the role **Earth Engine Resource Viewer**.
4. **Keys → Add Key → JSON** — download the file.
5. In the JSON file, copy the entire contents and minify to a single line (no newlines).
6. Paste that single-line JSON as the `GEE_SERVICE_ACCOUNT_KEY` environment variable in Railway (step 3).

The backend auto-detects the env var: if present it uses the service account; otherwise it falls back to application-default credentials (local dev).

---

### Step 3 — Backend on Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# From repo root
railway login
railway init        # link to a new or existing Railway project
railway up --service backend --source ./backend
```

Set these environment variables in the **Railway dashboard → Variables**:

| Variable | Value |
|---|---|
| `GEE_PROJECT` | Your GEE-enabled GCP project ID (e.g. `kelpwatch-2026`) |
| `GEE_SERVICE_ACCOUNT_KEY` | Full JSON of your service account key — single line, no newlines |
| `GROQ_API_KEY` | Your Groq key (free at console.groq.com) |
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_SERVICE_KEY` | `service_role` key from Supabase |
| `SUPABASE_JWT_SECRET` | JWT secret from Supabase API settings |
| `DAILY_QUERY_LIMIT` | `5` (or your preferred limit) |
| `LLM_PROVIDER` | `groq` (default) or `openai` |
| `OPENAI_API_KEY` | Required only if `LLM_PROVIDER=openai` |

The `railway.toml` at repo root already configures:
- Builder: Nixpacks
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `GET /api/status`
- Restart policy: on failure

After deploy, confirm the backend is live:

```bash
curl https://your-backend.railway.app/api/status
# → {"county_data_ready": true, "county_count": 6}
```

County data is precomputed from GEE in the background on startup — it takes ~60–90 seconds after cold boot before `county_data_ready` is `true`.

---

### Step 4 — Frontend on Vercel

**Update `frontend/config.js`** with production values before deploying:

```js
window.KELPWATCH_CONFIG = {
  apiBase:         'https://your-backend.railway.app/api',
  supabaseUrl:     'https://xxxx.supabase.co',
  supabaseAnonKey: 'eyJ...',   // anon key, safe to expose in frontend
};
```

```bash
# Install Vercel CLI
npm i -g vercel

# From repo root
vercel --prod
```

`vercel.json` is already configured to serve from `frontend/` with SPA rewrites.

---

## Data sources

| Domain | Layer | Dataset | Resolution |
|---|---|---|---|
| Kelp/eelgrass | Current (2022–2024) | Landsat 8/9 C02 (GEE) | 30 m |
| Kelp/eelgrass | Historical (1995–1997) | Landsat TM5 C02 (GEE) | 30 m |
| Forest fire | Burned area (last 30d) | MODIS MCD64A1 (GEE) | 500 m |
| India rainfall | Accumulated (last 30d) | UCSB-CHG/CHIRPS/DAILY (GEE) | ~5 km |
| India NDVI | Vegetation health (last 30d) | MODIS MOD13A2 (GEE) | 1 km |
| ESRP grants | Site locations + ROI | WDFW public data | — |
| 2040 goal | Restoration tracker | WA DNR / Puget Sound Partnership | — |

---

## Auth tiers

| Tier | Access |
|---|---|
| Unauthenticated | Map tiles + county panel + fire/India panels (read-only) |
| Free signed-up | All views + AI agent (5 queries/day, resets midnight UTC) |
| Enterprise | Unlimited — contact@empathyaitech.com |

---

## LLM agent

The `/api/agent` endpoint routes queries across three domains using keyword-based intent
classification before hitting the LLM, and warms a response cache for the 20 most common
questions on startup (after GEE county data is ready). Cache hits return instantly without
an LLM call.

Supported providers: **Groq** (default, free tier) and **OpenAI** (set `LLM_PROVIDER=openai`).

---

![KelpWatch](https://github.com/dpaul0501/kelpwatch/blob/main/kelpwatcpng.png)

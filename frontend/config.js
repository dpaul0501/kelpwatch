// Frontend is served by the FastAPI backend (same origin), so the API is at /api.
// Opening index.html directly from disk falls back to a local uvicorn on :8100.
// supabaseAnonKey is the publishable key — safe to expose in the browser.
window.KELPWATCH_CONFIG = {
  apiBase:         location.protocol === 'file:' ? 'http://localhost:8100/api' : '/api',
  supabaseUrl:     'https://jyutiduypnwoaorooitq.supabase.co',
  supabaseAnonKey: 'sb_publishable_HubkiLDQf-hGbugcwuxjpg_fq_QP-tw',
};

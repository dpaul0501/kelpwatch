# 🌊 KelpWatch — Puget Sound Restoration Intelligence

> Real satellite data. Real AI. Real impact. — HSI × WiT Regatta Hackathon 2026

## Live Demo
Open `frontend/index.html` in browser OR run the full stack below.

---

## Run in 3 commands

```bash
# 1. Clone and enter backend
cd kelpwatch/backend

# 2. Install deps
pip install -r requirements.txt

# 3. Set your keys in .env
# GEE_PROJECT=kelpwatch-2026
# ANTHROPIC_API_KEY=sk-ant-...

# 4. Run (GEE auth already done via `earthengine authenticate`)
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

---

## What's real

| Feature | Data Source |
|---|---|
| Current kelp layer | Landsat 8/9 C02 T1 L2 (2022–2024) via GEE |
| Historical baseline | Landsat TM5 C02 T1 L2 (1995–1997) via GEE |
| Change detection | NDWI per county, 4 time periods |
| ESRP grant sites | WDFW public data |
| ROI Agent | Anthropic Claude Haiku |
| 2040 goal tracker | WA DNR / Puget Sound Partnership |

## State Alignment
- WA DNR: 10,000 acres kelp/eelgrass by 2040
- ESRP 2026 grant round: $14.6M awarded
- Puget Sound Partnership Action Agenda
- Chinook / Southern Resident Orca ESA recovery

![KelpWatch](https://raw.githubusercontent.com/dpaul0501/kelpwatch/kelpwatchpng.png)

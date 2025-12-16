# AI-Driven Chiller Health Monitoring System (Realtime + Historical)

This package is your existing dashboard code (`rapp.txt`) upgraded to optionally fetch **Realtime chiller points** from:
`/ODataConnector/rest/RealtimeData?PointName=...`

## Folder contents
- `app.py` — Streamlit dashboard (updated)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit config
- `.streamlit/secrets.toml` — **put secrets here for Streamlit Cloud**
- `.env.example` — optional env template for local VS Code run

---

## 1) Run locally in VS Code (Windows) — tiny minute steps

### A. Create a clean folder
1. Create a folder: `C:\chiller_app`
2. Copy **all files** from this zip into `C:\chiller_app`

### B. Create a virtual environment
Open **PowerShell** in `C:\chiller_app`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### C. Add credentials
Option 1 (recommended): use `.streamlit/secrets.toml`
1. Open `C:\chiller_app\.streamlit\secrets.toml`
2. Set:
   - `REALTIME_USERNAME`
   - `REALTIME_PASSWORD`
   - (optional) `OPENAI_API_KEY`

Option 2: use environment variables
1. Copy `.env.example` to `.env`
2. Edit `.env` with your keys

### D. Start Streamlit
```powershell
streamlit run app.py --server.port 8501
```

If port is busy:
```powershell
streamlit run app.py --server.port 8504
```

Then open: `http://localhost:8501`

---

## 2) Deploy on Streamlit Cloud — tiny minute steps

### A. Put the code on GitHub
1. Create a new GitHub repo, e.g. `chiller-realtime-dashboard`
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - (do NOT commit `.streamlit/secrets.toml` with real passwords if repo is public)

### B. Create the Streamlit app
1. Go to Streamlit Cloud → **New app**
2. Select your repo
3. Main file: `app.py`

### C. Add secrets on Streamlit Cloud
1. In Streamlit Cloud → App → **Settings → Secrets**
2. Paste the content of `.streamlit/secrets.toml` (edit values)

### D. Run
Streamlit will build using `requirements.txt` and start.

---

## 3) How the Realtime API is integrated

- In the **left navigation panel** → section **Realtime**:
  - Enable **Use Realtime API**
  - Choose PointName template:
    - `dash` → `ac:T5/TR/CHILLER/T5-TR-CHILLER-{n}/{tag}`
    - `underscore` → `ac:T5/TR/CHILLER/T5_TR_CHILLER_{n}/{tag}`
  - Set refresh seconds

- The app fetches realtime points ONLY for **ON chillers** to keep calls smaller.

---

## Notes / Troubleshooting
- If you get `401 Unauthorized`: verify username/password in Secrets.
- If you get timeout / non-200: ensure Streamlit Cloud can reach the IP/port (many private DC IPs are not reachable from cloud).
  - In that case, run locally inside the same network/VPN as the BMS API.

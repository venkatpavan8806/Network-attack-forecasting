# Deploying the website (Supabase + Render + Vercel)

```
 your machine                         cloud
┌──────────────────────┐   HTTPS   ┌──────────────────────────┐    ┌─────────────────────┐
│ capture agent        │ ────────▶ │ Render: FastAPI backend  │ ─▶ │ Supabase Postgres   │
│ (scapy, Npcap/sudo)  │  features │  - verifies logins       │    │  per-user rows      │
│ 30 s window features │  + alerts │  - runs the LSTM, etc.   │    │  (RLS on, no public │
└──────────────────────┘           │  - pcap / CSV replay     │    │   API access)       │
                                   └──────────────────────────┘    └─────────────────────┘
 browser ── Vercel (React app) ── Supabase Auth (login) ──▲ access token on every API call
```

* **Every account starts empty.** All traffic, forecasts, alerts, packets and
  sensors are stored per user (`user_id` = Supabase user id) and every query
  filters on it. There is no shared demo dataset.
* **Packet capture runs in the agent, not the browser** — browsers cannot read
  network packets. Users download the agent from the *Capture* tab and run it
  on the machine they want to monitor. They can also upload `.pcap` files or
  generate fresh sample traffic.
* The models are trained **inside the Docker image** during each Render build
  (`python -m app.train`, ~10–15 min), from the project's own training pipeline.

You need free accounts on supabase.com, render.com and vercel.com, and this
repository on GitHub.

---

## 1. Supabase (database + login)

1. **New project**: pick a region near your Render region (e.g. Mumbai or
   Singapore), set a database password and keep it.
2. **Authentication → Sign In / Providers**
   * *Email*: enabled. For a demo you can turn **off** "Confirm email" so new
     accounts can sign in immediately.
   * *Anonymous sign-ins*: **enable** if you want the "Continue as guest" button.
3. **Copy these values:**
   | Value | Where | Used by |
   |---|---|---|
   | Project URL `https://<ref>.supabase.co` | Project Settings → Data API | Render `SUPABASE_URL`, Vercel `VITE_SUPABASE_URL` |
   | anon / publishable key | Project Settings → API Keys | Vercel `VITE_SUPABASE_ANON_KEY` |
   | **Session pooler** connection string | top bar **Connect** → *Session pooler* | Render `DATABASE_URL` |
   | JWT secret (only if your project uses the *legacy* JWT secret) | Project Settings → JWT Keys | Render `SUPABASE_JWT_SECRET` |

   Use the **Session pooler** string (host `aws-…pooler.supabase.com`, port
   5432), not the "Direct connection" — the direct host is IPv6-only and
   Render cannot reach it. Put your database password into the string.

You do **not** need to create tables: the backend creates them on first start
and enables Row Level Security on them (with no policies), so the public
Supabase REST API cannot read any of this data — only the backend can.

**Never** put the `service_role` key or the database password in the frontend.

## 2. Render (backend API)

1. Push this repository to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo. It reads
   [`render.yaml`](render.yaml) (Docker service `nadf-backend`, root `backend/`).
3. Fill in the environment variables it asks for:
   * `DATABASE_URL` — Supabase session-pooler string
   * `SUPABASE_URL` — Supabase project URL
   * `SUPABASE_JWT_SECRET` — leave empty unless your project is on the legacy secret
   * `CORS_ORIGINS` — your Vercel URL (you can fill this after step 3, e.g. `https://nadf.vercel.app`)
   * `PUBLIC_API_URL` — this service's URL, e.g. `https://nadf-backend.onrender.com` (optional)
4. Deploy. The first build installs PyTorch (CPU) and trains the models —
   expect **15–20 minutes**. When it is live, open
   `https://<your-service>.onrender.com/health`; it should show
   `"status": "ok", "auth_mode": "supabase", "database": "postgresql"`.

## 3. Vercel (frontend)

1. Vercel → **Add New → Project** → import the repo.
2. **Root Directory: `frontend`** (framework preset: Vite — [`frontend/vercel.json`](frontend/vercel.json) handles the rest).
3. Environment variables (see [`frontend/.env.example`](frontend/.env.example)):
   * `VITE_API_URL` = `https://<your-service>.onrender.com`
   * `VITE_SUPABASE_URL` = Supabase project URL
   * `VITE_SUPABASE_ANON_KEY` = Supabase anon / publishable key
4. Deploy, then:
   * Render → `nadf-backend` → Environment → set `CORS_ORIGINS` to the Vercel URL (comma-separate several, e.g. a custom domain).
   * Supabase → Authentication → **URL Configuration** → *Site URL* = the Vercel URL, and add it under *Redirect URLs* (used by email confirmation / password reset links).

## 4. Use it

1. Open the Vercel URL, create an account (or continue as guest). The workspace is empty.
2. **Capture → Add sensor** — copy the command with the token (shown once).
3. **Download agent (.zip)**, unzip, `pip install -r requirements.txt`, then:
   * Windows: install [Npcap](https://npcap.com), open an Administrator terminal, `python nadf_agent.py --token nadf_…`
   * Linux/macOS: `sudo python3 nadf_agent.py --token nadf_…`
4. The sensor turns **online** within ~30 s. Every remote host that opens
   connections *to* that machine appears as `live:<ip>` with a prediction after
   every 30-second window (from the first window). Try it with
   `nmap -sS <machine ip>` from another device on the network.
5. No second machine? Upload a Wireshark `.pcap`, or click **Generate sample data**.

## Limits of the free tiers (worth knowing before a demo)

* **Render free** sleeps after ~15 min without traffic; the next request takes
  ~1 minute to wake it. The website shows a "waking up" banner and retries; the
  agent queues windows and retries. Open the site a minute before a demo.
  512 MB RAM: the API uses ~350 MB (measured with CPU PyTorch).
* **Supabase free** pauses a project after a week without activity — resume it from the dashboard.
* Max upload size is 25 MB (`MAX_UPLOAD_MB`).

## Running locally (no cloud accounts needed)

```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m app.train
venv\Scripts\python -m uvicorn app.api.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

Without `SUPABASE_URL` / `VITE_SUPABASE_URL` the app runs in **dev auth mode**:
no login, SQLite storage, and every browser gets its own random workspace id.
Run the agent locally with `python nadf_agent.py --token nadf_… --server http://127.0.0.1:8000`.
To run the tests against Postgres instead of SQLite, set `TEST_DATABASE_URL`.

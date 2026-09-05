# xPts frontend

React and Tailwind frontend for exploring the xPts projection API. The interface covers player projections, fixture forecasts, full simulation distributions, inference metadata, and the API contract without relying on player or club imagery.

## Local development

Run the Flask API from the repository root:

```powershell
.\venv\Scripts\python.exe -m backend.src.api
```

Then run the frontend in a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Vite serves the app at `http://127.0.0.1:5173` and proxies `/api` and `/health` to `http://127.0.0.1:5000` by default.

## API configuration

Configuration is build-time and lives in an untracked `.env.local` or deployment environment:

```dotenv
VITE_API_BASE_URL=https://api.example.com
```

Leave `VITE_API_BASE_URL` empty for a same-origin deployment. `VITE_DEV_API_PROXY` only controls the local Vite proxy. A separate production API origin must allow the frontend origin through its CORS policy.

## Checks and production build

```powershell
npm run typecheck
npm run build
npm run preview
```

The production assets are written to `frontend/dist`.

# Deployment

Two pieces: a React front end on Vercel, and the FastAPI backend on Google
Cloud Run. The backend needs ~700 MB of memory to hold Whisper and the intent
classifier, which is why it is not on a serverless or 512 MB free tier.

## 1. Backend - Google Cloud Run

One-time setup:

```bash
gcloud auth login                       # opens a browser
gcloud projects create voicebot-slp     # or reuse an existing project id
gcloud config set project voicebot-slp
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

Billing must be enabled on the project (Cloud Run requires a billing account on
file). The free tier covers 2 million requests and 360,000 GB-seconds a month,
which is far more than a demo uses.

Deploy:

```bash
gcloud run deploy voicebot-api \
  --source . \
  --region asia-south1 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --concurrency 4 \
  --min-instances 0 \
  --allow-unauthenticated
```

`--min-instances 0` lets it scale to zero so idle time is free; the cost is a
cold start of about 12 seconds while the container loads both models (measured
locally on the same image). The
front end pings `/health` on page load so that happens while the visitor is
still reading.

The command prints a service URL like `https://voicebot-api-xxxx.run.app`.

## 1b. Backend - Railway (alternative to Cloud Run)

Railway builds the same Dockerfile. `railway.json` pins the builder so it does
not try to autodetect Node from `web/`, and the healthcheck points at `/health`
with a 300s timeout because the container loads ~700 MB of models before it
answers.

```bash
npm install -g @railway/cli
railway login
railway init            # create the project
railway up              # build and deploy from this directory
railway domain          # assign a public *.up.railway.app URL
```

Railway has no free tier: it gives a one-time $5 trial credit, then Hobby is
$5/month which includes $5 of usage. This service idles at roughly 360 MB, so
leaving it running costs a few dollars a month - enable **App Sleeping** in the
service settings so it suspends when idle and the credit lasts.

Measured on this image: ~12s from cold start to serving, ~2.4s per voice
request, 358 MB under load. Give the service at least 1 GB.

## 2. Front end - Vercel

```bash
cd web
vercel login
vercel link                                      # create/link the project
vercel env add VITE_API_URL production           # paste the Cloud Run URL
vercel deploy --prod
```

`VITE_API_URL` must be the backend URL (Cloud Run or Railway) with no trailing
slash. It is
read at build time, so changing it requires a redeploy.

## 3. Lock down CORS (optional)

The API allows any origin by default. Once the Vercel URL is known:

```bash
gcloud run services update voicebot-api \
  --region asia-south1 \
  --set-env-vars ALLOWED_ORIGINS=https://<your-app>.vercel.app
```

## Local development

```bash
pip install -r requirements-api.txt
cd app && uvicorn main:app --port 8080      # backend

cd web && npm install && npm run dev        # front end on :5173, proxies to :8080
```

## Alternative front end

`streamlit_app.py` runs the same pipeline as a single-file Streamlit app
(`pip install -r requirements-streamlit.txt`, then `streamlit run streamlit_app.py`).

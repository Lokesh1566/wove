# Wove

**An AI stylist that builds outfits from the clothes you already own.**

You own more clothes than you wear. Wove photographs your closet, learns every
piece, and assembles real outfits from what is already hanging there, so you
shop less and wear more of what you have.

> Working name. Rename freely.

---

## The problem

Wardrobes keep growing while the set of outfits people actually reach for stays
flat. Roughly 80% of a closet gets worn 20% of the time. The cost is money,
decision fatigue every morning, and a lot of fabric that never gets used. Most
styling apps answer this by selling you more. Wove starts from your real
wardrobe and only suggests a purchase when one versatile piece would genuinely
multiply what you can wear.

## What it does

- **Tags every item.** Upload a photo, get back dominant colors and category.
  Color analysis runs in the browser. Category is a quick tap, or optional
  in-browser CLIP auto-tagging.
- **Builds outfits.** Pairs items across slots (top, bottom, shoes, layer) and
  scores each combination by real color harmony rules.
- **Stays useful.** Occasion-aware vibe filter, plus a gap finder that names the
  single most valuable piece to add.

The web app in `frontend/` is a complete, self-contained product: it runs
entirely in the browser, needs no backend, and deploys free on GitHub Pages.
It includes a company landing site, an account system (sign up, sign in, log
out) with a private per-user closet, 50+ men's and women's garment types,
filtering, color analysis, the outfit engine, and the gap finder.

> Accounts and closets are stored in the browser for the prototype. For real
> cross-device login, swap the local auth layer for Supabase, Firebase, or the
> FastAPI backend with a database. The auth code is isolated to make this swap
> small.

The FastAPI service in `backend/` is the optional server-side path for heavier
CLIP tagging and embeddings at wardrobe scale.

---

## Architecture

```
                 ┌──────────────┐        multipart image
   Browser  ───▶ │  Frontend    │ ───────────────────────┐
   (React /      │  (static)    │                         ▼
    static)      └──────────────┘               ┌───────────────────┐
        ▲                                        │   FastAPI API     │
        │   outfits, tags (JSON)                 │  /tag  /outfits   │
        └────────────────────────────────────── │  /health          │
                                                 └─────────┬─────────┘
                                                           │
                                      ┌────────────────────┼───────────────────┐
                                      ▼                     ▼                   ▼
                               CLIP (ViT-B-32)      color extraction     outfit engine
                               image + text          (PIL median cut)    (cosine score,
                               embeddings                                 FAISS-ready)
```

The frontend is fully static and can sit on GitHub Pages or Vercel. The API is
a Dockerized FastAPI service that runs anywhere that runs Python (Hugging Face
Spaces, Render, Fly, a VM). They talk over plain JSON.

## Tech stack

| Layer    | Choice                                                    |
| -------- | --------------------------------------------------------- |
| Frontend | HTML / CSS / vanilla JS today, React + Tailwind next      |
| API      | FastAPI, Dockerized                                       |
| Tagging  | OpenCLIP ViT-B-32 zero-shot (category + style)            |
| Color    | PIL median-cut quantization                               |
| Pairing  | Cosine similarity over embeddings, FAISS-ready            |

---

## Repo structure

```
wove/
├── frontend/
│   └── index.html          # landing + product demo (the live face of Wove)
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI app: /health, /tag, /outfits
│   │   ├── tagging.py      # CLIP zero-shot + deterministic mock fallback
│   │   ├── colors.py       # dominant color extraction
│   │   ├── outfits.py      # slot pairing + similarity scoring
│   │   └── schemas.py      # pydantic models
│   ├── requirements.txt    # core deps, API boots with just these
│   ├── requirements-ml.txt # torch + open_clip + faiss for real tagging
│   └── Dockerfile
└── ml/
    └── notes.md            # model notes and next steps
```

---

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API boots immediately on the core requirements using a deterministic mock
tagger, so every endpoint responds out of the box. Open
`http://127.0.0.1:8000/docs` for the interactive API.

Try it:

```bash
curl -F "file=@/path/to/shirt.jpg;type=image/jpeg" http://127.0.0.1:8000/tag
```

To switch on real CLIP tagging:

```bash
pip install -r requirements-ml.txt
```

The model downloads on first call, then `/tag` returns true CLIP embeddings and
zero-shot categories. The response field `backend` tells you which path served
the request (`mock` or `clip:ViT-B-32`).

### Frontend (the complete app)

No build step. Open `frontend/index.html` directly, or serve it:

```bash
cd frontend
python -m http.server 5173
# open http://127.0.0.1:5173
```

Upload photos of your clothes, tap a category for each (or hit "Auto-tag with
AI"), and Wove builds scored outfits from what you own. Everything runs client
side, so this is also exactly what deploys to GitHub Pages.

---

## Roadmap

- [x] In-browser color extraction
- [x] Outfit pairing with color harmony + formality scoring
- [x] Complete client-side web app: upload, wardrobe, outfits, gap finder
- [x] Weather-aware and occasion-aware styling (morning, lunch, office, dinner, party, night)
- [x] Styled outfit preview (the look assembled on a figure)
- [x] Two-step login with email code (prototype shows the code; production emails it)
- [x] Background cleanup for clean product-style closet images
- [x] Optional in-browser CLIP auto-tagging
- [x] FastAPI backend with CLIP + mock fallback (optional server path)
- [ ] Real cross-device auth + email OTP (Supabase or backend)
- [ ] Style calendar: plan and track what you wear
- [ ] Style stats: wear counts and cost per wear
- [ ] Wishlist and shop-with-your-closet
- [ ] Real virtual try-on (server-side generative model)
- [ ] Learned compatibility model to replace the heuristic scorer

## Deploy notes

GitHub Pages serves the static frontend only, it cannot run the Python API.
Recommended split: frontend on Pages or Vercel, API on Hugging Face Spaces
(Docker) or Render. Lock CORS to your deployed frontend origin before launch.

---

Built by Lokesh Reddy Elluri · [github.com/Lokesh1566](https://github.com/Lokesh1566)

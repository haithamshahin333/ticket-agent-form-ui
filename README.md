# underwriting-pipeline

A **multi-node LangGraph pipeline with a coupled web UI**, served by the deployment itself and
driven through its own **Runs API**. Submit a structured application; four specialized nodes
(**intake → risk → pricing → decision**) analyze it over shared state and return a structured
underwriting recommendation. Deliberately **not a chatbot** — each stage is its own graph node and
its own span in LangSmith. Deployable on LangSmith Deployments; scaffolded with the LangGraph CLI.

> Demo domain is **commercial-property underwriting** with **synthetic** data — no real customer
> data. The *shape* is the point: structured input → a sequence of specialized nodes over shared
> state → deterministic tools do the math, the model adds narrative → a structured report. Swap in
> another domain by editing `domain.py` + `tools.py`; the graph, UI, and deployment stay the same.

---

## Deploy to LangSmith (`langgraph deploy`)

Exactly what to do to put this on a LangSmith Deployment.

### Prerequisites
- A **LangSmith account** and API key (create one at https://smith.langchain.com → Settings → API Keys).
- The **LangGraph CLI**: `uv pip install "langgraph-cli"` (or `pipx install langgraph-cli`).
- No local Docker needed — the CLI builds **remotely** when Docker isn't available.

### 1. Put your model credential in `.env`
`langgraph deploy` reads `langgraph.json`'s `env` (here, `.env`) and uploads its **non-reserved**
values as deployment **secrets**. For this app the only required one is the model key:

```bash
cp .env.example .env
```
Then edit `.env`:
```dotenv
OPENAI_API_KEY=sk-...            # REQUIRED — uploaded as a deployment secret
# OPENAI_BASE_URL=https://...    # optional — only if routing through an OpenAI-compatible gateway
# MODEL=openai:gpt-5-mini        # optional — this is the default

LANGSMITH_API_KEY=lsv2_...       # authenticates the deploy call (NOT uploaded as a runtime secret)
```

### 2. Deploy
Run from the project root (where `langgraph.json` lives):

```bash
langgraph deploy --name underwriting-pipeline --deployment-type dev
```
- Builds the image (remotely if Docker isn't running locally) and **creates or updates** the
  LangSmith Deployment named `underwriting-pipeline`.
- Uploads `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` / `MODEL` if set) as **secrets**.
- Authenticates with `LANGSMITH_API_KEY` from `.env` (or pass `--api-key`, or set
  `LANGSMITH_DEPLOYMENT_NAME` in `.env` instead of `--name`).
- Use `--deployment-type prod` for a production deployment; add `--no-wait` to skip waiting.

### 3. Open the UI
Grab the deployment URL from the CLI output (or `langgraph deploy list`) and open:

```
https://<your-deployment>.us.langgraph.app/app
```
Click **Load sample application**, paste your **LangSmith API key** into the key field
(**required on a deployment** — see [Auth](#auth-on-a-deployment)), then **Analyze application**.
Watch the four nodes complete in turn and the recommendation render.

### What the platform manages for you — do NOT set these
| Concern | Handled by | Notes |
|---------|-----------|-------|
| **Tracing** | Platform | `LANGSMITH_TRACING`, the tracing key, and a project named after the deployment are injected automatically. `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` are **reserved** (not uploaded as runtime vars). Every node appears as its own span — no tracing code in this repo. |
| **Persistence** | Platform | Postgres is auto-provisioned; this app has **no checkpointer in code** on purpose, and runs persist automatically by `thread_id`. `POSTGRES_URI` / `REDIS_URI` are reserved. |
| **Run API + streaming** | Platform | `POST /threads`, `/threads/{id}/runs/stream`, `/runs/wait` are added on the same origin. The UI calls them directly — no server code to enqueue jobs or poll status. |

### Manage the deployment
```bash
langgraph deploy list                          # find your deployment + URL
langgraph deploy logs                          # tail logs
langgraph deploy --name underwriting-pipeline  # re-run to ship an update
langgraph deploy delete                        # tear it down
```

> Prefer the UI? You can also create the deployment from the LangSmith **Deployments** tab by
> connecting a GitHub repo and setting the same env vars there (mark `OPENAI_API_KEY` as a secret).

### Auth on a deployment
The `/app` page loads without a credential (the `http.app` route is unauthenticated by default —
leave `enable_custom_route_auth` **off** so a browser can open it). But the **Runs API calls it
makes are gated by `x-api-key`**, so a credential is **required to actually run the pipeline** on a
deployment — the key field is optional only for local `langgraph dev` (no auth). Without a key the
page loads but **Analyze** fails with 401/403.

The key you paste is **workspace-scoped** and kept in memory for the tab only (not stored) — fine
for a demo **you** (or each attendee) drive, but do **not** hand out a page carrying *your* key to
untrusted users. The production-correct answer is custom auth: validate a per-user OIDC/JWT bearer
token in a `langgraph.json` `auth` handler (sent as an `Authorization: Bearer …` header) so no
privileged key ever reaches the browser.

---

## Run locally

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
cd underwriting-pipeline
uv venv && source .venv/bin/activate
uv pip install -e .                      # install this project's deps
uv pip install "langgraph-cli[inmem]"    # the dev server

cp .env.example .env                     # set OPENAI_API_KEY (+ OPENAI_BASE_URL if using a gateway)
langgraph dev                            # starts the agent server on http://localhost:2024
```

Open **http://localhost:2024/app**, click **Load sample application**, then **Analyze application**.
Leave the API-key field blank — the local dev server needs no auth. Bump the prior-loss figures up
and re-run to watch the decision move from **quote** → **refer** → **decline**.

Run the tool tests (pure functions — no model, no network):
```bash
uv run --group dev pytest         # or: PYTHONPATH=src pytest
```

## How it works

```
browser form (static/index.html)
  └─ fetch → Runs API (same origin: POST /threads, POST /threads/{id}/runs/stream)
                                             └─ "agent" graph (StateGraph)
                                                  intake → risk → pricing → decision   (shared UnderwritingState)
                                                    └─ pure tools (tools.py) + one LLM synthesis per node
```

- A LangGraph **`StateGraph`**: four nodes over a shared `UnderwritingState`. Each node reads the
  previous node's result from state and writes its own — `intake` validates + derives exposure,
  `risk` scores it, `pricing` builds the premium, `decision` makes the quote/refer/decline call and
  assembles the report.
- **Functional core, imperative shell.** All the real logic — validation, risk scoring, premium
  build-up, the decision rule — is **pure functions in `tools.py`**, unit-tested with no model and
  no network. Nodes are thin wrappers that call those tools, then ask the model for a one-to-three
  sentence narrative *on top of* the already-computed numbers. The deterministic result is always
  the source of truth.
- The UI **streams** the run via `POST /threads/{id}/runs/stream` with
  `stream_mode: ["updates","values"]`: each `updates` event lights up the corresponding node and
  shows its commentary live; the final `values` snapshot drives the structured report. Falls back to
  the blocking `/runs/wait` endpoint if streaming isn't available.
- **What you don't build:** tracing, persistence, and the run/stream API are the platform's, not
  yours — no custom trace logger, no in-memory session store, no background-task/poll plumbing.
- **Hardening carried over:** applicant free-text (`business_name`, `notes`) is passed to the model
  only inside explicit data-boundary tags (never as instructions); the UI renders **every** dynamic
  value with `textContent` (never `innerHTML`); the API key is sent only as the `x-api-key` header
  and is never form-serialized, stored, or logged.
- A **model factory** (`src/agent/model.py`) — the single place to swap models. Constructed lazily,
  so importing the graph/tools never requires credentials.

### Layout
```
src/agent/
├── domain.py          # types + synthetic underwriting reference data (hazard / CAT / construction / appetite)
├── tools.py           # pure functions: validate_application → compute_exposure → score_risk → price_policy → decide
├── graph.py           # StateGraph: intake → risk → pricing → decision -> `graph`
├── model.py           # get_model() — change the LLM here
├── webapp.py          # Starlette app serving the UI at /app (langgraph.json http.app)
└── static/index.html  # single-file UI: application form → live per-node progress → structured report
tests/test_tools.py    # unit tests for the pure tools (no model / no network)
langgraph.json         # graphs.agent + http.app
```

### Change the model
Edit `get_model()` in `src/agent/model.py`. The default uses `init_chat_model` with `MODEL`
(e.g. `openai:gpt-5-mini`) and honors `OPENAI_BASE_URL`. A commented Claude-via-LangSmith-Gateway
alternative sits right beside it.

### Change the domain
Edit `domain.py` (the input `Application` shape + reference constants) and the pure functions in
`tools.py`. The graph, streaming UI, and deployment don't change — only the form fields in
`static/index.html` and the report labels need to match your new fields.

> **Why `fetch` instead of `@langchain/langgraph-sdk`?** The JS SDK is designed for a bundler
> (Vite/webpack). Loaded straight from an ESM CDN it fails at import (its `langsmith` dependency
> throws `does not provide an export named '__version__'`), which silently breaks the page. For a
> build-free single-file demo, calling the Runs API with `fetch` (including parsing the SSE stream)
> is the reliable, minimal choice. If you want the SDK's `Client` object, add a bundler step and
> `import { Client } from "@langchain/langgraph-sdk"`.

## Out of scope (kept minimal on purpose)

Explicit checkpointer/store config, OIDC/SSO, conditional/branching routes, human-in-the-loop
pauses, a saved run-history UI. The `StateGraph` gives you the seams to add any of these later.

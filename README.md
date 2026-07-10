# ticket-agent-ui

A **minimal LangGraph agent with a coupled web UI** (a form: **ticket ID** + **chat box**),
served by the deployment itself and triggered through its own **Runs API**. Deployable on
LangSmith Deployments; scaffolded with the LangGraph CLI.

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
langgraph deploy --name ticket-agent-ui --deployment-type dev
```
- Builds the image (remotely if Docker isn't running locally) and **creates or updates** the
  LangSmith Deployment named `ticket-agent-ui`.
- Uploads `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` / `MODEL` if set) as **secrets**.
- Authenticates with `LANGSMITH_API_KEY` from `.env` (or pass `--api-key`, or set
  `LANGSMITH_DEPLOYMENT_NAME` in `.env` instead of `--name`).
- Use `--deployment-type prod` for a production deployment; add `--no-wait` to skip waiting.

### 3. Open the UI
Grab the deployment URL from the CLI output (or `langgraph deploy list`) and open:

```
https://<your-deployment>.us.langgraph.app/app
```
Enter a ticket ID (`TICKET-101` / `TICKET-102`) and a message, paste your **LangSmith API key**
into the key field (**required on a deployment** — see [Auth](#auth-on-a-deployment)), and click **Run**.

### What the platform manages for you — do NOT set these
| Concern | Handled by | Notes |
|---------|-----------|-------|
| **Tracing** | Platform | `LANGSMITH_TRACING`, the tracing key, and a project named after the deployment are injected automatically. `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` are **reserved** (not uploaded as runtime vars). |
| **Persistence** | Platform | Postgres is auto-provisioned; this app has **no checkpointer in code** on purpose, and threads persist automatically by `thread_id`. `POSTGRES_URI` / `REDIS_URI` are reserved. |

### Manage the deployment
```bash
langgraph deploy list                 # find your deployment + URL
langgraph deploy logs                 # tail logs
langgraph deploy --name ticket-agent-ui   # re-run to ship an update
langgraph deploy delete               # tear it down
```

> Prefer the UI? You can also create the deployment from the LangSmith **Deployments** tab by
> connecting a GitHub repo and setting the same env vars there (mark `OPENAI_API_KEY` as a secret).

### Auth on a deployment
The `/app` page loads without a credential (the `http.app` route is unauthenticated by default —
leave `enable_custom_route_auth` **off** so a browser can open it). But the **Runs API calls it
makes are gated by `x-api-key`**, so a credential is **required to actually run the agent** on a
deployment — the key field is optional only for local `langgraph dev` (no auth). Without a key the
page loads but **Run** fails with 401/403.

The key you paste is **workspace-scoped** and kept in memory for the tab only (not stored) — fine
for a demo **you** (or each attendee) drive, but do **not** hand out a page carrying *your* key to
untrusted users. The production-correct answer is custom auth: validate a per-user OIDC/JWT bearer
token in a `langgraph.json` `auth` handler (sent as an `Authorization: Bearer …` header) so no
privileged key ever reaches the browser.

---

## Run locally

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
cd ticket-agent-ui
uv venv && source .venv/bin/activate
uv pip install -e .                      # install this project's deps
uv pip install "langgraph-cli[inmem]"    # the dev server

cp .env.example .env                     # set OPENAI_API_KEY (+ OPENAI_BASE_URL if using a gateway)
langgraph dev                            # starts the agent server on http://localhost:2024
```

Open **http://localhost:2024/app**, enter a ticket ID and message, click **Run**. Leave the
API-key field blank — the local dev server needs no auth.

## How it works

```
browser form (static/index.html)
  └─ fetch → Runs API (same origin: POST /threads, POST /threads/{id}/runs/wait)
                                             └─ "agent" graph (create_agent)
                                                  └─ get_ticket tool
```

- LangChain v1 `create_agent` with a custom state field (`ticket_id`) supplied as run input.
- A `@dynamic_prompt` middleware that builds the system prompt from the ticket id.
- A UI **coupled to the deployment** via `http.app`, driving the agent through the same-origin
  Runs API — the same REST endpoints the LangGraph SDK wraps, called directly with `fetch` so the
  page has **zero JS dependencies and no build step** (see note below).
- Multi-turn conversation: one thread per browser tab (persistence is provided by the server).
- A **model factory** (`src/agent/model.py`) — the single place to swap models.

### Layout
```
src/agent/
├── graph.py            # the agent: TicketState + get_ticket tool + dynamic prompt -> `graph`
├── model.py            # get_model() — change the LLM here
├── webapp.py           # Starlette app serving the UI at /app (langgraph.json http.app)
└── static/index.html   # the single-file form UI (plain fetch to the Runs API, no deps)
langgraph.json          # graphs.agent + http.app
```

### Change the model
Edit `get_model()` in `src/agent/model.py`. The default uses `init_chat_model` with `MODEL`
(e.g. `openai:gpt-5-mini`) and honors `OPENAI_BASE_URL`. A commented Claude-via-LangSmith-Gateway
alternative sits right beside it.

> **Why `fetch` instead of `@langchain/langgraph-sdk`?** The JS SDK is designed for a bundler
> (Vite/webpack). Loaded straight from an ESM CDN it fails at import (its `langsmith` dependency
> throws `does not provide an export named '__version__'`), which silently breaks the page. For a
> build-free single-file demo, calling the Runs API with `fetch` is the reliable, minimal choice.
> If you want the SDK's `Client` object, add a bundler step and `import { Client } from "@langchain/langgraph-sdk"`.

## Out of scope (kept minimal on purpose)

Streaming, explicit checkpointer/store config, OIDC/SSO, thread history UI, tests.

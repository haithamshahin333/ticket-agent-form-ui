"""Custom HTTP app served by the LangGraph deployment (`http.app` in langgraph.json).

Serves the single-file UI at GET /app. The built-in Runs/Threads API is added by the
platform on the same origin, so the page can talk to it with no cross-origin setup.
"""

from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route

_INDEX = Path(__file__).parent / "static" / "index.html"


async def ui(request):  # noqa: ANN001, ANN201, ARG001
    """Serve the ticket-agent form UI."""
    # no-store so browsers always fetch the current page (avoids serving a stale cached UI).
    return FileResponse(_INDEX, headers={"Cache-Control": "no-store"})


app = Starlette(routes=[Route("/app", ui)])

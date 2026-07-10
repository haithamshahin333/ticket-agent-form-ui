"""Model factory — the single place to change which LLM the agent uses.

Default: an OpenAI-compatible model via `init_chat_model`, honoring `OPENAI_BASE_URL`
so calls can route through a workshop / gateway endpoint. Swap the provider by editing
`get_model()` (a commented Anthropic-via-LangSmith-gateway alternative is included).
"""

import os
from typing import Any

from langchain.chat_models import init_chat_model


def get_model() -> Any:
    """Return the chat model the agent runs on.

    Change models here in one place. `MODEL` uses an `init_chat_model` id
    (e.g. ``openai:gpt-5-mini``); `OPENAI_BASE_URL`, if set, routes OpenAI-compatible
    traffic through a custom gateway. `OPENAI_API_KEY` is read from the environment.
    """
    model_id = os.environ.get("MODEL", "openai:gpt-5-mini")
    kwargs: dict[str, Any] = {}

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(model_id, **kwargs)

    # --- Alternative: Claude via the LangSmith LLM Gateway -----------------------
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(
    #     model=os.environ.get("MODEL", "claude-sonnet-4-5-20250929"),
    #     base_url="https://gateway.smith.langchain.com/anthropic",
    #     api_key=os.environ["LANGSMITH_API_KEY"],
    # )

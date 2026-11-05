"""Ciclo de vida de SQLite e invocaciones con traza observable."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from agente.config import Settings
from agente.graph import AgentState, build_graph

logger = logging.getLogger(__name__)
Graph = CompiledStateGraph[AgentState, Any, Any, Any]


@asynccontextmanager
async def open_agent(
    settings: Settings, model: BaseChatModel | None = None
) -> AsyncIterator[Graph]:
    settings.database.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        model = ChatOpenAI(model=settings.model, timeout=30, max_retries=2)
    async with AsyncSqliteSaver.from_conn_string(str(settings.database)) as checkpointer:
        yield build_graph(model, checkpointer, settings.max_context_messages)


async def run_turn(
    graph: Graph, prompt: str, thread_id: str, recursion_limit: int = 10
) -> dict[str, Any]:
    if not thread_id.strip() or not prompt.strip():
        raise ValueError("El prompt y thread_id son obligatorios")
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    events: list[dict[str, Any]] = [{"tipo": "usuario", "contenido": prompt}]
    answer: str | list[str | dict[str, Any]] = ""
    async for update in graph.astream(
        {"messages": [{"role": "user", "content": prompt}]}, config, stream_mode="updates"
    ):
        for node, data in update.items():
            for message in data.get("messages", []):
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        events.append({"tipo": "acciones", "llamadas": message.tool_calls})
                    else:
                        answer = message.content
                        events.append({"tipo": "respuesta", "contenido": answer})
                elif isinstance(message, ToolMessage):
                    events.append(
                        {
                            "tipo": "observacion",
                            "herramienta": message.name,
                            "llamada_id": message.tool_call_id,
                            "estado": message.status,
                            "contenido": message.content,
                        }
                    )
            logger.info("Nodo completado: %s", node)
    return {"thread_id": thread_id, "eventos": events, "respuesta": answer}


def save_trace(path: Path, payload: dict[str, Any]) -> None:
    """Guardar solo trazas de datos ficticios; pueden contener prompts y resultados."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

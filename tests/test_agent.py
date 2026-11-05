"""Pruebas del grafo real con respuestas del LLM controladas, sin API ni costos."""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agente.config import Settings
from agente.graph import context_to_keep
from agente.runtime import open_agent, run_turn, save_trace
from agente.tools import buscar_pedidos


class ScriptedModel(FakeMessagesListChatModel):
    """Doble SOLO de pruebas; no toma decisiones ni se usa en la CLI productiva."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def action(cursor: int = 0, client: int = 102) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "buscar_pedidos",
                "args": {"cliente_id": client, "cursor": cursor},
                "id": f"call-{client}-{cursor}",
                "type": "tool_call",
            }
        ],
    )


async def test_multistep_and_persistence(tmp_path: Path) -> None:
    settings = Settings(database=tmp_path / "memory.sqlite")
    model = ScriptedModel(
        responses=[
            action(),
            action(2),
            AIMessage(content="El cliente 102 tuvo 3 pedidos por ARS 14.500."),
        ]
    )
    async with open_agent(settings, model) as graph:
        first = await run_turn(
            graph, "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?", "demo"
        )
        observations = [e for e in first["eventos"] if e["tipo"] == "observacion"]
        assert len(observations) == 2
        pages = [json.loads(e["contenido"]) for e in observations]
        assert pages[0]["completo"] is False
        assert pages[1]["completo"] is True
        orders = [order for page in pages for order in page["pedidos"]]
        assert len(orders) == 3
        assert sum(order["total_centavos"] for order in orders) == 1_450_000

    # Se cierra y reabre la conexión: la memoria no depende de la instancia del grafo.
    async with open_agent(
        settings, ScriptedModel(responses=[AIMessage(content="Último: 1003.")])
    ) as graph:
        saved = await graph.aget_state({"configurable": {"thread_id": "demo"}})
        assert any("cliente 102" in str(m.content) for m in saved.values["messages"])
        assert len([m for m in saved.values["messages"] if isinstance(m, ToolMessage)]) == 2
        second = await run_turn(graph, "¿y el último?", "demo")
        state = await graph.aget_state({"configurable": {"thread_id": "demo"}})
        assert state.values["messages"][-2].content == "¿y el último?"
        assert state.values["messages"][0].content == first["eventos"][0]["contenido"]
        isolated = await graph.aget_state({"configurable": {"thread_id": "otro"}})
        assert not isolated.values

    # Artefacto generado por una ejecución real del grafo con modelo simulado.
    save_trace(tmp_path / "trace.json", {"origen": "modelo_simulado", "turnos": [first, second]})
    assert (
        json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))["origen"]
        == "modelo_simulado"
    )


async def test_recoverable_error_returns_to_model(tmp_path: Path) -> None:
    model = ScriptedModel(
        responses=[action(99), action(), action(2), AIMessage(content="3 pedidos")]
    )
    async with open_agent(Settings(database=tmp_path / "m.sqlite"), model) as graph:
        result = await run_turn(graph, "Pedidos del cliente 102", "error")
    observations = [e for e in result["eventos"] if e["tipo"] == "observacion"]
    assert json.loads(observations[0]["contenido"])["error"] == "cursor_fuera_de_rango"
    assert len(observations) == 3
    assert result["respuesta"] == "3 pedidos"


async def test_unknown_client_can_request_clarification(tmp_path: Path) -> None:
    model = ScriptedModel(
        responses=[action(client=999), AIMessage(content="¿Cuál es el ID correcto?")]
    )
    async with open_agent(Settings(database=tmp_path / "m.sqlite"), model) as graph:
        result = await run_turn(graph, "Cliente 999", "unknown")
    assert "ID correcto" in result["respuesta"]
    assert "cliente_no_encontrado" in result["eventos"][2]["contenido"]


async def test_recursion_limit(tmp_path: Path) -> None:
    async with open_agent(
        Settings(database=tmp_path / "m.sqlite"),
        ScriptedModel(responses=[action() for _ in range(10)]),
    ) as graph:
        with pytest.raises(GraphRecursionError):
            await run_turn(graph, "Pedidos", "loop", recursion_limit=4)


async def test_tool_validation_and_empty_results() -> None:
    with pytest.raises(ValueError):
        await buscar_pedidos.ainvoke({"cliente_id": -1})
    result = await buscar_pedidos.ainvoke({"cliente_id": 103})
    assert result["pedidos"] == []
    assert result["completo"] is True


def test_context_preserves_complete_turns() -> None:
    messages = [
        HumanMessage(content="viejo"),
        AIMessage(content="respuesta"),
        HumanMessage(content="nuevo"),
        action(),
        ToolMessage(content="parcial", tool_call_id="call-102-0"),
        action(2),
        ToolMessage(content="completo", tool_call_id="call-102-2"),
    ]
    kept = context_to_keep(messages, 4)
    assert kept == messages[2:]
    assert kept[0].type == "human"


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        Settings(recursion_limit=0)
    with pytest.raises(ValueError):
        Settings(max_context_messages=1)

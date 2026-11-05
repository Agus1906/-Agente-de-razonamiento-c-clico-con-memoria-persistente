"""Genera evidencia reproducible del grafo, NO evidencia de autonomía de un LLM real."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agente.config import Settings
from agente.runtime import open_agent, run_turn, save_trace


class DemoModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "DemoModel":
        return self


def call(cursor: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "buscar_pedidos",
                "args": {"cliente_id": 102, "cursor": cursor},
                "id": f"demo-{cursor}",
                "type": "tool_call",
            }
        ],
    )


async def main() -> None:
    with TemporaryDirectory() as directory:
        settings = Settings(database=Path(directory) / "demo.sqlite")
        model = DemoModel(
            responses=[
                call(0),
                call(2),
                AIMessage(content="El cliente 102 tuvo 3 pedidos por ARS 14.500."),
            ]
        )
        async with open_agent(settings, model) as graph:
            first = await run_turn(
                graph, "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?", "demo-102"
            )
        followup = DemoModel(
            responses=[
                AIMessage(content="El último fue el pedido 1003, del 28/10/2026, por ARS 4.000.")
            ]
        )
        async with open_agent(settings, followup) as graph:
            state = await graph.aget_state({"configurable": {"thread_id": "demo-102"}})
            assert len(state.values["messages"]) == 6
            second = await run_turn(graph, "¿y el último?", "demo-102")
        save_trace(
            Path("traces/ejemplo-offline.json"),
            {
                "origen": "modelo_simulado_grafo_y_herramientas_reales",
                "advertencia": "Decisiones guionadas; no valida autonomía OpenAI.",
                "sqlite_reabierto_entre_turnos": True,
                "turnos": [first, second],
            },
        )
    print("Traza generada: traces/ejemplo-offline.json")


if __name__ == "__main__":
    asyncio.run(main())

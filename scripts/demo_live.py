"""Prueba de aceptación con OpenAI real: exige >=2 llamadas y comprueba memoria."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import ToolMessage

from agente.config import Settings
from agente.runtime import open_agent, run_turn, save_trace


async def main() -> None:
    base = Settings.from_env()
    with TemporaryDirectory() as directory:
        settings = Settings(model=base.model, database=Path(directory) / "acceptance.sqlite")
        async with open_agent(settings) as graph:
            first = await run_turn(
                graph, "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?", "aceptacion"
            )
        observations = [e for e in first["eventos"] if e["tipo"] == "observacion"]
        if len(observations) < 2:
            raise AssertionError("El LLM no completó dos llamadas; revisar traza y prompt")
        pages = [json.loads(event["contenido"]) for event in observations]
        orders = {order["pedido_id"]: order for page in pages for order in page.get("pedidos", [])}
        assert len(orders) == 3
        assert sum(order["total_centavos"] for order in orders.values()) == 1_450_000
        async with open_agent(settings) as graph:
            state = await graph.aget_state({"configurable": {"thread_id": "aceptacion"}})
            assert len([m for m in state.values["messages"] if isinstance(m, ToolMessage)]) >= 2
            second = await run_turn(graph, "¿y el último?", "aceptacion")
        save_trace(
            Path("traces/local-openai.json"),
            {
                "origen": "openai_real",
                "modelo": settings.model,
                "sqlite_reabierto_entre_turnos": True,
                "turnos": [first, second],
                "validacion": "Revisar respuestas: 3 pedidos, ARS 14.500; último 1003, ARS 4.000.",
            },
        )
        print(first["respuesta"])
        print(second["respuesta"])
        print("Revisá traces/local-openai.json antes de copiarlo a traces/ejemplo-openai.json.")


if __name__ == "__main__":
    asyncio.run(main())

"""CLI asíncrona de una consulta por ejecución; thread_id conserva la sesión."""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from langgraph.errors import GraphRecursionError

from agente.config import Settings
from agente.runtime import open_agent, run_turn, save_trace


async def execute(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Configurá OPENAI_API_KEY en .env o en el entorno")
    async with open_agent(settings) as graph:
        result = await run_turn(graph, args.prompt, args.thread_id, settings.recursion_limit)
    if args.trace:
        save_trace(Path(args.trace), {"origen": "openai_real", **result})
    print(result["respuesta"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente ReAct con memoria SQLite")
    parser.add_argument("prompt")
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--trace", help="Ruta JSON; usar datos ficticios")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(execute(args))
    except GraphRecursionError:
        parser.exit(2, "Se alcanzó el límite de pasos. Reformulá o iniciá un nuevo thread_id.\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"No se pudo ejecutar: {exc}\n")
    except Exception:
        # No imprimir excepciones del proveedor: podrían incluir información sensible.
        parser.exit(
            1, "Falló el proveedor o la ejecución. Revisá conexión, modelo y credenciales.\n"
        )


if __name__ == "__main__":
    main()

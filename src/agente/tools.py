"""Base de datos simulada, de solo lectura, con importes exactos en centavos."""

from typing import Annotated, TypedDict

from langchain_core.tools import tool
from pydantic import Field

PositiveId = Annotated[int, Field(strict=True, gt=0)]
Cursor = Annotated[int, Field(strict=True, ge=0)]


class Order(TypedDict):
    pedido_id: int
    fecha: str
    total_centavos: int


ORDERS: dict[int, list[Order]] = {
    102: [
        {"pedido_id": 1001, "fecha": "2026-10-01", "total_centavos": 450000},
        {"pedido_id": 1002, "fecha": "2026-10-15", "total_centavos": 600000},
        {"pedido_id": 1003, "fecha": "2026-10-28", "total_centavos": 400000},
    ],
    103: [],
}


@tool
async def buscar_pedidos(cliente_id: PositiveId, cursor: Cursor = 0) -> dict[str, object]:
    """Consulta los pedidos ficticios de un cliente por ID positivo, en páginas de dos.

    Usar para contar pedidos, sumar importes o identificar el último pedido de un cliente.
    Empezar con cursor=0. Si completo=false, los datos son PARCIALES: volver a llamar
    con el mismo cliente_id y el siguiente_cursor recibido antes de calcular totales.
    Los importes están en centavos ARS y las fechas en ISO, orden ascendente.
    Si devuelve cliente_no_encontrado, solicitar al usuario el ID correcto; no inventarlo.
    Un cliente existente sin pedidos devuelve una lista vacía con completo=true.
    Esta consulta es de solo lectura y se puede reintentar sin duplicar operaciones.
    """
    if cliente_id not in ORDERS:
        return {"error": "cliente_no_encontrado", "cliente_id": cliente_id, "reintentable": False}
    orders = ORDERS[cliente_id]
    if cursor > len(orders):
        return {"error": "cursor_fuera_de_rango", "reintentable": True, "cursor_sugerido": 0}
    end = cursor + 2
    complete = end >= len(orders)
    return {
        "cliente_id": cliente_id,
        "pedidos": orders[cursor:end],
        "moneda": "ARS",
        "completo": complete,
        "siguiente_cursor": None if complete else end,
    }


TOOLS = [buscar_pedidos]

# Validación de la entrega

Fecha: 05/11/2026.

Entorno local: Windows, Python 3.12.15, dependencias en `requirements-lock.txt`.

| Comprobación | Resultado |
|---|---|
| `pytest -q` | 7 pruebas aprobadas |
| `ruff check .` | Sin errores |
| `mypy src` (strict) | Sin errores en los 6 módulos |
| `python scripts/demo_offline.py` | Traza JSON generada |
| Dos consultas a herramienta en el mismo turno | Verificado con grafo real y LLM simulado |
| Reabrir SQLite con el mismo thread_id | Verificado |
| Aislamiento de otro thread_id | Verificado |
| Retorno tras cursor inválido | Verificado con LLM simulado |
| Cliente desconocido y aclaración | Verificado con LLM simulado |
| Límite duro de recursión | Verificado |
| Autonomía y respuesta del proveedor real | Pendiente de API key propia y `demo_live.py` |

La traza offline no se presenta como una ejecución de OpenAI. La aceptación
integral requiere revisar una ejecución real según los pasos del README.

"""Configuración validada; ninguna credencial se incluye en logs."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-4.1-mini"
    database: Path = Path("data/checkpoints.sqlite")
    recursion_limit: int = 10
    max_context_messages: int = 40

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OPENAI_MODEL no puede estar vacío")
        if not 2 <= self.recursion_limit <= 100:
            raise ValueError("RECURSION_LIMIT debe estar entre 2 y 100")
        if self.max_context_messages < 4:
            raise ValueError("MAX_CONTEXT_MESSAGES debe ser al menos 4")

    @classmethod
    def from_env(cls) -> "Settings":
        # No buscar .env en carpetas superiores de otros proyectos.
        load_dotenv(Path.cwd() / ".env")
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            database=Path(os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite")),
            recursion_limit=int(os.getenv("RECURSION_LIMIT", "10")),
            max_context_messages=int(os.getenv("MAX_CONTEXT_MESSAGES", "40")),
        )

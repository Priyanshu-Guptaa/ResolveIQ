"""Domain layer: framework-agnostic models shared by every engine.

Nothing in this package imports FastAPI, SQLAlchemy, or ChromaDB -- it is
pure Python/Pydantic so it can be unit tested in isolation and reused by
any future interface (CLI, batch job, different UI).
"""

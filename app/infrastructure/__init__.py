"""Infrastructure layer: persistence and external system clients.

Domain and engine code depend on abstractions (repository/store Protocols);
concrete implementations here are the only place that know about SQLAlchemy
or ChromaDB specifics.
"""

"""One parser per file family. Each implements the FileParser Protocol
(app.engines.ingestion.file_type_registry) -- can_parse(filename) and
parse(filename, content) -> ParsedFile. None of them know about each
other or about zip recursion; that orchestration lives in the registry.
"""

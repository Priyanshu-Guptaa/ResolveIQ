"""Evidence Ingestion Pipeline: file-type detection -> the correct parser
-> normalized text, so entity extraction and the Recommendation Engine
never see raw bytes (RFC rev 1, §04).
"""

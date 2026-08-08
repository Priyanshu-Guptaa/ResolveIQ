"""The one place the ``KnowledgeObjectType`` display labels live on
the UI side (Sprint 3, Phase 3.4) -- ``object_picker.py`` and
``10_Relationship_Manager.py`` each carried their own copy of this dict
since Phase 3.3; centralizing it here is the same "avoid duplicated ...
logic" the framework's backend applies, just for a UI constant.
"""

from __future__ import annotations

TYPE_LABELS: dict[str, str] = {
    "component": "Component",
    "document": "Document",
    "known_bug": "Known Bug",
    "sql_template": "SQL Template",
    "historical_investigation": "Historical Investigation",
    "playbook": "Playbook",
    "product": "Product",
    "technology": "Technology",
    "version": "Version",
    "log_source_application": "Log Source Application",
    "log_collection_scenario": "Log Collection Scenario",
}
OBJECT_TYPES: list[str] = list(TYPE_LABELS.keys())

MANUAL_CREATE_UNSUPPORTED: set[str] = {"log_source_application", "log_collection_scenario"}
"""Types whose required fields (a nested LogRepositoryLocation; no
single title/name field at all) don't fit the generic quick-create
form below -- these are created by Log Wiki Import instead. Browsing,
editing, and lifecycle management still work identically to every
other type once a record exists."""

"""Application-wide logging setup.

Call :func:`configure_logging` once at process startup (API startup event,
Streamlit boot, or a script's ``__main__``). Every module then uses
``logging.getLogger(__name__)`` as usual.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported under a test runner) --
        # avoid duplicate handlers/log lines.
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party libraries are noisy at INFO/DEBUG; keep them quieter
    # unless the whole app is running in DEBUG.
    if level.upper() != "DEBUG":
        for noisy in ("chromadb", "sentence_transformers", "httpx", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # Known cosmetic issue: on some chromadb/posthog version combinations,
    # anonymized_telemetry=False doesn't stop chromadb from *attempting* a
    # telemetry call, and the failed attempt logs at ERROR ("capture() takes
    # 1 positional argument but 3 were given"). It's non-fatal and unrelated
    # to application behavior -- silenced explicitly rather than masking
    # real chromadb errors by lowering the whole logger's level.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

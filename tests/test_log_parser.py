"""Unit tests for the generic multi-format log parser."""

from __future__ import annotations

from app.domain.enums import LogLevel
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor
from app.engines.log_intelligence.log_parser import GenericLogParser

SAMPLE_JAVA_LOG = """2026-08-05 14:22:01,123 INFO [http-nio-8080-exec-7] com.acme.order.OrderController - Received checkout request
2026-08-05 14:22:01,345 ERROR [http-nio-8080-exec-7] com.acme.order.OrderProcessor - Failed to calculate total
java.lang.NullPointerException: shippingAddress is null
    at com.acme.order.OrderProcessor.calculateTotal(OrderProcessor.java:142)
    at com.acme.order.OrderController.checkout(OrderController.java:58)
2026-08-05 14:22:01,346 ERROR [http-nio-8080-exec-7] com.acme.order.OrderController - checkout failed with HTTP 500
"""


def _parser() -> GenericLogParser:
    return GenericLogParser(RegexEntityExtractor())


def test_parses_one_event_per_top_level_line():
    events = _parser().parse(SAMPLE_JAVA_LOG)
    # 3 log lines + 1 exception header line = 4 top-level events; the two
    # "at ..." frames fold into the exception header's event.
    assert len(events) == 4


def test_stack_trace_frames_fold_into_preceding_event():
    events = _parser().parse(SAMPLE_JAVA_LOG)
    exception_event = next(e for e in events if "NullPointerException" in e.message)

    assert "OrderProcessor.calculateTotal" in exception_event.message
    assert "OrderController.checkout" in exception_event.message


def test_detects_timestamp_and_level():
    events = _parser().parse(SAMPLE_JAVA_LOG)
    first = events[0]

    assert first.timestamp is not None
    assert first.timestamp.year == 2026
    assert first.level == LogLevel.INFO
    assert events[1].level == LogLevel.ERROR


def test_detects_bracketed_component():
    events = _parser().parse(SAMPLE_JAVA_LOG)
    assert events[0].source_component == "http-nio-8080-exec-7"


def test_events_carry_extracted_entities():
    log = "2026-08-05 03:12:01 ERROR CollectorService - CommandTimeout for meter=80071234567 endpoint_id=EP-99213"
    events = _parser().parse(log)

    assert len(events) == 1
    entity_values = [e.value for e in events[0].entities]
    assert "80071234567" in entity_values
    assert "EP-99213" in entity_values


def test_empty_text_returns_no_events():
    assert _parser().parse("") == []

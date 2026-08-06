"""Unit tests for the regex-based entity extractor.

Covers entity types across the application/infra/meter-comm domains, not
just meter numbers, per the platform requirement that ResolveIQ not assume
every investigation is meter-related.
"""

from __future__ import annotations

from app.domain.enums import EntityType
from app.engines.log_intelligence.entity_extractor import RegexEntityExtractor


def _values_for(entities, entity_type: EntityType) -> list[str]:
    return [e.value for e in entities if e.entity_type == entity_type]


def test_extracts_correlation_and_request_ids():
    extractor = RegexEntityExtractor()
    text = "Received request request_id=REQ-88213 correlation_id=CORR-9f31ac user_id=cust-4471"

    entities = extractor.extract(text)

    assert "REQ-88213" in _values_for(entities, EntityType.REQUEST_ID)
    assert "CORR-9f31ac" in _values_for(entities, EntityType.CORRELATION_ID)
    assert "cust-4471" in _values_for(entities, EntityType.USER_ID)


def test_extracts_exception_type_and_stack_trace_frames():
    extractor = RegexEntityExtractor()
    text = (
        "java.lang.NullPointerException: shippingAddress is null\n"
        "    at com.acme.order.OrderProcessor.calculateTotal(OrderProcessor.java:142)\n"
        "    at com.acme.order.OrderController.checkout(OrderController.java:58)\n"
    )

    entities = extractor.extract(text)

    assert "java.lang.NullPointerException" in _values_for(entities, EntityType.EXCEPTION_TYPE)
    stack_frames = _values_for(entities, EntityType.STACK_TRACE)
    assert any("OrderProcessor.calculateTotal" in frame for frame in stack_frames)


def test_extracts_meter_and_serial_and_endpoint():
    extractor = RegexEntityExtractor()
    text = "CommandTimeout for meter=80071234567 endpoint_id=EP-99213 serial_number=SN-4471882"

    entities = extractor.extract(text)

    assert "80071234567" in _values_for(entities, EntityType.METER_NUMBER)
    assert "EP-99213" in _values_for(entities, EntityType.ENDPOINT_ID)
    assert "SN-4471882" in _values_for(entities, EntityType.SERIAL_NUMBER)


def test_extracts_ip_address_and_host_and_url():
    extractor = RegexEntityExtractor()
    text = "Call to https://payments.internal.acme.com/api/v1/charge failed, host=order-svc-01, client 10.12.4.31"

    entities = extractor.extract(text)

    assert "https://payments.internal.acme.com/api/v1/charge" in _values_for(entities, EntityType.URL)
    assert "order-svc-01" in _values_for(entities, EntityType.HOST_NAME)
    assert "10.12.4.31" in _values_for(entities, EntityType.IP_ADDRESS)


def test_extracts_kafka_topic_and_consumer_group_and_sql_session():
    extractor = RegexEntityExtractor()
    text = "consumer_group=billing-events-consumer-group topic=billing.events.v1 spid=61"

    entities = extractor.extract(text)

    assert "billing-events-consumer-group" in _values_for(entities, EntityType.CONSUMER_GROUP)
    assert "billing.events.v1" in _values_for(entities, EntityType.KAFKA_TOPIC)
    assert "61" in _values_for(entities, EntityType.SQL_SESSION)


def test_deduplicates_repeated_entities():
    extractor = RegexEntityExtractor()
    text = "correlation_id=CORR-1 ... correlation_id=CORR-1 again"

    entities = extractor.extract(text)

    assert len(_values_for(entities, EntityType.CORRELATION_ID)) == 1


def test_empty_text_returns_no_entities():
    extractor = RegexEntityExtractor()
    assert extractor.extract("") == []

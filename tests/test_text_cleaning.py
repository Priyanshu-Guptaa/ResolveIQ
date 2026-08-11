"""Tests for strip_low_signal_boilerplate -- see the module's own
docstring for the real live-verified case (a ServiceNow task-template
disclaimer + empty header fields inflating local KB similarity scores)
that motivated this."""

from app.engines.shared.text_cleaning import strip_low_signal_boilerplate


def test_empty_and_none_are_noop():
    assert strip_low_signal_boilerplate("") == ""
    assert strip_low_signal_boilerplate(None) is None


def test_strips_the_known_disclaimer_paragraph():
    text = (
        'This Template is expected to be reviewed in full. If the template is not followed, '
        'the task may be rejected.\n'
        'Overview of Task Creation Process for S3G teams per product is documented in detail in KB0017312\n\n'
        'If the issue going to a Devices or Firmware team please use the template – "Metering Case L2, L3 Task Template"\n\n'
        '1. Technology: RF Mesh\n\n'
        '2. Defect Description:\n'
        'Customer reports meter program changes fail once the meter reaches final Normal status.'
    )
    cleaned = strip_low_signal_boilerplate(text)

    assert "This Template is expected to be reviewed" not in cleaned
    assert "KB0017312" not in cleaned
    assert "Customer reports meter program changes fail" in cleaned
    assert "Technology: RF Mesh" in cleaned


def test_strips_empty_label_lines_but_keeps_lines_with_a_value():
    text = (
        "1. Organization Name: Arizona Public Service Company\n"
        "Priority: Medium\n"
        "Technology:  RF Mesh\n"
        "Last Upgrade Date: \n"
        "Environment Setup: \n"
        "Browser: (Chrome, Firefox, Edge, IE)\n"
        "GridStream Integration Suite:\n"
        "Defect Description: Meter program change event failed after upgrade."
    )
    cleaned = strip_low_signal_boilerplate(text)

    assert "Organization Name: Arizona Public Service Company" in cleaned
    assert "Priority: Medium" in cleaned
    assert "Technology:  RF Mesh" in cleaned
    assert "Browser: (Chrome, Firefox, Edge, IE)" in cleaned
    assert "Defect Description: Meter program change event failed after upgrade." in cleaned
    # Empty-value lines are gone.
    assert "Last Upgrade Date:" not in cleaned
    assert "Environment Setup:" not in cleaned
    assert "GridStream Integration Suite:" not in cleaned


def test_never_strips_a_line_with_a_placeholder_value():
    """Deliberate boundary: a line like 'Last Upgrade Date: 1/1/0000' has
    *a* value, even a low-quality placeholder one -- stripping it would
    require guessing at every possible 'this doesn't really count'
    pattern across real customer data, which this fix explicitly avoids."""
    text = "Last Upgrade Date: 1/1/0000\nPatches: 2"
    cleaned = strip_low_signal_boilerplate(text)

    assert "Last Upgrade Date: 1/1/0000" in cleaned
    assert "Patches: 2" in cleaned


def test_two_real_documents_sharing_the_disclaimer_no_longer_look_identical_at_the_head():
    """Regression test for the actual live-found bug: two DIFFERENT real
    defects that both happened to open with the identical disclaimer
    paragraph (which is what inflated their similarity score to 86%)
    should no longer share that identical prefix once cleaned."""
    shared_prefix = (
        'This Template is expected to be reviewed in full. If the template is not followed, '
        'the task may be rejected.\n'
        'Overview of Task Creation Process for S3G teams per product is documented in detail in KB0017312\n\n'
        'If the issue going to a Devices or Firmware team please use the template – "Metering Case L2, L3 Task Template"\n\n'
    )
    doc_a = shared_prefix + "1. Technology: RF Mesh\n\n2. Defect Description:\nMeter program change fails at final Normal status."
    doc_b = shared_prefix + "1. Technology: RF Mesh\n\n2. Defect Description:\nCustomer requests a full meter read analysis."

    cleaned_a = strip_low_signal_boilerplate(doc_a)
    cleaned_b = strip_low_signal_boilerplate(doc_b)

    assert cleaned_a != cleaned_b
    assert "meter program change fails" in cleaned_a.lower()
    assert "full meter read analysis" in cleaned_b.lower()
    assert "This Template" not in cleaned_a
    assert "This Template" not in cleaned_b

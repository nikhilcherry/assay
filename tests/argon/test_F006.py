"""Regression test for F006: compile_rule time window number words 'twelve' and 'twenty-four'."""

from assay.r2.rule_compiler import compile_rule


def test_compile_rule_twelve_and_twenty_four_hour_windows():
    spec_24_hyphen = compile_rule(
        "ADV-4. Test. Three or more cards sharing one device profile within twenty-four hours: recommend CREATE_CASE."
    )
    assert spec_24_hyphen.window_seconds == 24 * 3600

    spec_24_space = compile_rule(
        "ADV-5. Test. Three or more cards sharing one device profile within twenty four hours: recommend CREATE_CASE."
    )
    assert spec_24_space.window_seconds == 24 * 3600

    spec_12 = compile_rule(
        "ADV-6. Test. Three or more cards sharing one device profile within twelve hours: recommend CREATE_CASE."
    )
    assert spec_12.window_seconds == 12 * 3600

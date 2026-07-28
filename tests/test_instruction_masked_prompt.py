from __future__ import annotations

from edit2forensics.pilot_prompt import format_user_turn


def test_masked_prompt_contains_no_instruction_text():
    messages = format_user_turn(
        object(), object(), "SECRET INSTRUCTION", include_instruction=False,
    )
    text = messages[0]["content"][2]["text"]
    assert "SECRET INSTRUCTION" not in text
    assert "intentionally withheld" in text


def test_unmasked_prompt_retains_instruction_text():
    messages = format_user_turn(
        object(), object(), "VISIBLE INSTRUCTION", include_instruction=True,
    )
    text = messages[0]["content"][2]["text"]
    assert 'Instruction: "VISIBLE INSTRUCTION"' in text

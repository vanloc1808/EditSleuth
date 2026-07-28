"""Shared prompt construction for pilot training and evaluation."""
from __future__ import annotations


def format_user_turn(
    real_image,
    edited_image,
    instruction: str,
    include_instruction: bool = True,
) -> list[dict]:
    if include_instruction:
        prompt = (
            "You are a forensic image-edit detector. Given a "
            "(real, edited) image pair and the edit instruction, "
            "produce a structured analysis of the edit.\n\n"
            f'Instruction: "{instruction}"'
        )
    else:
        prompt = (
            "You are a forensic image-edit detector. Given only a "
            "(real, edited) image pair, produce a structured analysis "
            "of the edit. The edit instruction is intentionally withheld."
        )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": real_image},
            {"type": "image", "image": edited_image},
            {"type": "text", "text": prompt},
        ],
    }]

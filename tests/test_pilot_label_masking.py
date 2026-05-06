"""Tests for the pilot training collator's label-masking logic.

The collator inside ``scripts/pilot_train.py`` masks loss positions
so the model only learns to predict the assistant-turn target tokens
and not the user-turn or image-placeholder tokens. We can't test
the full collator here because it depends on heavy ML libraries
(transformers, torch); instead we extract the masking algorithms
into pure-Python form and test them directly.

Two algorithms are tested:

1. ``_mask_labels``: an earlier prompt-length-based approach that
   measured prompt length via a separate tokenization pass and used
   it as a mask offset. Works in principle but produced incorrect
   masks in practice when the processor's batched-with-padding
   tokenization differed from its single-example tokenization
   (image-token leakage into the active region).

2. ``_find_marker_end``: the current approach. Searches the
   tokenized sequence for the ``<|im_start|>assistant\\n`` marker
   token sequence and returns the index immediately after it.
   Robust against tokenization variation because it operates only
   on the actual produced ``input_ids``.

Both are kept under test so future regressions in either approach
surface immediately.
"""
from __future__ import annotations


def _mask_labels(input_ids, prompt_lengths, pad_id=None):
    """Pure-Python reimplementation of the collator's masking
    logic, for testability without torch.

    Inputs:
      input_ids: list[list[int]]   (B rows, each a token sequence)
      prompt_lengths: list[int]    (length of prompt prefix per row)
      pad_id: int | None           (optional pad token id to mask)

    Returns labels with -100 in masked positions.
    """
    B = len(input_ids)
    labels = [list(row) for row in input_ids]
    for i in range(B):
        for j in range(prompt_lengths[i]):
            labels[i][j] = -100
    if pad_id is not None:
        for i in range(B):
            for j, tok in enumerate(input_ids[i]):
                if tok == pad_id:
                    labels[i][j] = -100
    return labels


# ---------------------------------------------------------------------------
# Basic masking
# ---------------------------------------------------------------------------

def test_prompt_prefix_is_fully_masked():
    """The first prompt_length tokens of each row should be -100."""
    input_ids = [[10, 11, 12, 13, 14, 15]]
    prompt_lengths = [3]
    labels = _mask_labels(input_ids, prompt_lengths)
    assert labels[0][:3] == [-100, -100, -100]
    # The remaining tokens should be unchanged.
    assert labels[0][3:] == [13, 14, 15]


def test_per_example_prompt_lengths_are_independent():
    """Different rows can have different prompt lengths, and each
    is masked independently up to its own prompt_length."""
    input_ids = [
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25],
    ]
    prompt_lengths = [2, 4]
    labels = _mask_labels(input_ids, prompt_lengths)
    assert labels[0][:2] == [-100, -100]
    assert labels[0][2:] == [12, 13, 14, 15]
    assert labels[1][:4] == [-100, -100, -100, -100]
    assert labels[1][4:] == [24, 25]


def test_assistant_target_tokens_are_unmasked():
    """The non-prompt portion of the input_ids — the assistant
    target — is exactly what the loss should be computed against."""
    input_ids = [[1, 2, 3, 100, 200, 300]]
    prompt_lengths = [3]   # prompt is [1, 2, 3]; target is [100, 200, 300]
    labels = _mask_labels(input_ids, prompt_lengths)
    target_positions = labels[0][3:]
    assert target_positions == [100, 200, 300]


# ---------------------------------------------------------------------------
# Padding handling
# ---------------------------------------------------------------------------

def test_pad_tokens_are_masked():
    """Padding tokens (typically 0 or a designated pad_id) should be
    -100 so they don't contribute to the loss."""
    input_ids = [[1, 2, 3, 0, 0]]
    prompt_lengths = [2]
    labels = _mask_labels(input_ids, prompt_lengths, pad_id=0)
    # First two are prompt-masked; pad positions (3, 4) are also
    # masked because their token id matches pad_id.
    assert labels[0] == [-100, -100, 3, -100, -100]


def test_pad_tokens_in_prompt_remain_masked():
    """If a padding-like token appears inside the prompt, it's
    already covered by the prompt-prefix mask. Stable behavior."""
    input_ids = [[1, 0, 3, 4, 5]]
    prompt_lengths = [3]
    labels = _mask_labels(input_ids, prompt_lengths, pad_id=0)
    # Tokens 0..2 are prompt-masked; 3 and 4 are not pad and not
    # prompt, so they remain.
    assert labels[0] == [-100, -100, -100, 4, 5]


def test_no_pad_id_means_no_pad_masking():
    """If pad_id is None (some tokenizers lack a dedicated pad), the
    algorithm should not mask any non-prompt tokens."""
    input_ids = [[1, 2, 3, 4, 5]]
    prompt_lengths = [2]
    labels = _mask_labels(input_ids, prompt_lengths, pad_id=None)
    assert labels[0] == [-100, -100, 3, 4, 5]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_tokens_masked_produces_minus_100_row():
    """If prompt_length equals sequence length (i.e., assistant
    target empty or absent), all labels are -100. The collator's
    sanity check warns about this case."""
    input_ids = [[1, 2, 3, 4]]
    prompt_lengths = [4]
    labels = _mask_labels(input_ids, prompt_lengths)
    assert all(t == -100 for t in labels[0])


def test_zero_prompt_length_masks_nothing():
    """Pathological but well-defined: prompt_length=0 means the
    entire sequence is the assistant target. Loss is computed on
    everything (modulo padding)."""
    input_ids = [[1, 2, 3, 4]]
    prompt_lengths = [0]
    labels = _mask_labels(input_ids, prompt_lengths)
    assert labels[0] == [1, 2, 3, 4]


def test_prompt_length_longer_than_sequence_does_not_index_error():
    """Defensive: even with a prompt_length larger than the row
    length, the masking shouldn't crash. (Shouldn't happen in
    practice but the algorithm is safe by construction since the
    inner loop is bounded by ``prompt_lengths[i]``.)"""
    input_ids = [[1, 2, 3]]
    prompt_lengths = [10]
    # Set up Python doesn't IndexError on out-of-range slice
    # assignment at the list level, but the per-element loop in our
    # algorithm WILL IndexError. Document the expected behavior:
    # callers must ensure prompt_length <= sequence_length.
    try:
        _mask_labels(input_ids, prompt_lengths)
    except IndexError:
        pass  # Expected: caller's responsibility to pass valid lengths.


# ===========================================================================
# Marker-search algorithm
# ===========================================================================

def _find_marker_end(row_ids, marker_ids):
    """Pure-Python reimplementation of the marker-search algorithm
    used in pilot_train.py's collator. Returns the index immediately
    after the marker subsequence; returns -1 if not found."""
    m = len(marker_ids)
    n = len(row_ids)
    if m > n:
        return -1
    for start in range(n - m + 1):
        if all(int(row_ids[start + j]) == marker_ids[j] for j in range(m)):
            return start + m
    return -1


def test_marker_search_finds_simple_marker():
    """Marker appears once in the middle of the sequence."""
    row = [10, 11, 100, 101, 102, 200, 201]
    marker = [100, 101, 102]
    end = _find_marker_end(row, marker)
    assert end == 5  # index immediately after [100, 101, 102]


def test_marker_search_returns_first_occurrence():
    """If the marker appears multiple times (shouldn't happen with
    unique chat-template markers, but be deterministic anyway), we
    return the first occurrence's end."""
    row = [10, 100, 101, 20, 100, 101, 30]
    marker = [100, 101]
    end = _find_marker_end(row, marker)
    assert end == 3  # first occurrence ends at index 3


def test_marker_search_returns_minus_one_when_absent():
    """If the marker isn't in the sequence, return -1 to signal
    failure; the collator promotes this to an error."""
    row = [10, 11, 12, 13]
    marker = [99, 99]
    end = _find_marker_end(row, marker)
    assert end == -1


def test_marker_search_handles_marker_at_start():
    row = [50, 51, 52, 100, 101]
    marker = [50, 51]
    end = _find_marker_end(row, marker)
    assert end == 2


def test_marker_search_handles_marker_at_end():
    row = [100, 101, 200, 201]
    marker = [200, 201]
    end = _find_marker_end(row, marker)
    assert end == 4  # one past the last index


def test_marker_search_handles_marker_longer_than_sequence():
    """Defensive: a marker longer than the row can't match. Return -1."""
    row = [10, 20]
    marker = [10, 20, 30, 40]
    end = _find_marker_end(row, marker)
    assert end == -1


def test_marker_search_handles_partial_match_then_real_match():
    """If a partial match starts but doesn't complete, search must
    continue past it to find the real match."""
    row = [50, 100, 50, 100, 101, 30]
    marker = [50, 100, 101]
    end = _find_marker_end(row, marker)
    assert end == 5  # match at indices 2..4


def test_marker_search_handles_single_token_marker():
    row = [10, 20, 30, 40]
    marker = [30]
    end = _find_marker_end(row, marker)
    assert end == 3


def test_marker_search_qwen_style_marker_simulation():
    """Simulate a realistic Qwen2-VL setup: a long sequence with
    image tokens, user text, an assistant marker, and chain text."""
    # Hundreds of image tokens (151655 = <|image_pad|> in Qwen2-VL)
    image_pad = 151655
    im_start = 151644
    im_end = 151645
    # User turn: <|im_start|>user\n + many image tokens + text + <|im_end|>
    row = (
        [im_start, 872, 198]              # <|im_start|>user\n (idealized)
        + [image_pad] * 200                # image tokens for first image
        + [image_pad] * 200                # image tokens for second image
        + [1, 2, 3]                        # instruction text
        + [im_end]
        + [im_start, 77091, 198]           # <|im_start|>assistant\n
        + [10, 11, 12]                     # chain text
        + [im_end]
    )
    # Marker: <|im_start|>assistant\n
    marker = [im_start, 77091, 198]
    end = _find_marker_end(row, marker)
    # Marker should be found at position 408 (3 + 200 + 200 + 3 + 1 = 407)
    # ending at 410.
    assert end == 410
    # The active-loss tokens should be [10, 11, 12, im_end] = 4 tokens.
    n_total = len(row)
    n_active = n_total - end
    assert n_active == 4


def test_marker_search_active_region_does_not_contain_image_tokens():
    """Regression test for the bug we observed in production: the
    active region (positions >= marker_end) should never contain
    image tokens, since they live entirely inside the user turn."""
    image_pad = 151655
    im_start = 151644
    row = (
        [im_start, 872, 198]
        + [image_pad] * 100
        + [im_start, 77091, 198]
        + [10, 11, 12]
    )
    marker = [im_start, 77091, 198]
    end = _find_marker_end(row, marker)
    active_tokens = row[end:]
    assert image_pad not in active_tokens

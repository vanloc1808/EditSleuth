"""Stage E: ReasoningAnnotator.

Composes one ``ReasoningArtifact`` per triplet from the upstream
artifacts produced by Stages A-D. The output is a structured header
plus a numbered-prose reasoning chain that becomes the supervised
target for downstream VLM training.

What's grounded vs what's prior
-------------------------------
The chain interleaves two kinds of statements:

* **Triplet-grounded** statements: claims about *this triplet* derived
  from the upstream artifacts. The instruction text, the spatial
  descriptor of the mask, the structural-change score, the difficulty
  bin. These are computed, not hand-curated.
* **Category-level priors**: the "what to look for" forensic-signature
  statement is a hand-curated, category-conditional fact from
  ``templates.py``. The chain phrases these with hedging language
  ("typically," "often") to flag them as priors rather than per-
  triplet observations.

This separation matters for faithfulness: a forensic detector trained
on these chains learns to ground its observations in actual evidence
(mask geometry, magnitude scores) while drawing on category priors
for what to check.

Why no LLM
----------
Chain composition is template-driven, not generative. Every statement
in a chain comes either from a structured artifact field or a fixed
template string. This is fast (~ms per chain), free, deterministic,
and easy to audit. If the templates are wrong, fix the templates;
no per-chain debugging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from edit2forensics.data.category_artifact import CategoryArtifact
from edit2forensics.data.difficulty_artifact_v2 import DifficultyArtifactV2
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.reasoning_artifact import ReasoningArtifact
from edit2forensics.data.triplet import EditTriplet
from edit2forensics.reasoning.spatial import (
    SPATIAL_DESCRIPTOR_PROSE,
    compute_spatial_descriptor,
)
from edit2forensics.reasoning.templates import (
    TEMPLATE_VERSION,
    get_forensic_prior,
)

log = logging.getLogger(__name__)


@dataclass
class ReasoningAnnotatorConfig:
    """Tuning knobs for chain composition."""

    central_margin: float = 0.25
    """Used in spatial descriptor: fraction of image around the center
    that counts as ``centered`` rather than a quadrant."""

    scattered_threshold: float = 0.3
    """Used in spatial descriptor: largest connected component must be
    at least this fraction of total mask area to be ``single-component``."""

    chain_template_version: str = TEMPLATE_VERSION


class ReasoningAnnotator:
    """Compose reasoning chains from upstream artifacts.

    Stateless. Instantiate once per process. ``annotate()`` is the
    main entry point, taking one record's worth of artifacts and
    returning a ``ReasoningArtifact``.
    """

    def __init__(
        self,
        config: ReasoningAnnotatorConfig | None = None,
    ) -> None:
        self.config = config or ReasoningAnnotatorConfig()

    def annotate(
        self,
        triplet: EditTriplet,
        mask_artifact: MaskArtifact,
        difficulty_artifact: DifficultyArtifactV2,
        category_artifact: CategoryArtifact,
    ) -> ReasoningArtifact:
        # ---- compute spatial descriptor (Stage E's only "new" signal) ----
        spatial = compute_spatial_descriptor(
            mask_artifact.mask_path,
            edit_scope=mask_artifact.edit_scope,
            central_margin=self.config.central_margin,
            scattered_threshold=self.config.scattered_threshold,
        )

        # ---- build header (structured, one-line) -------------------------
        header = (
            f"[category={category_artifact.category}, "
            f"scope={mask_artifact.edit_scope}, "
            f"difficulty={difficulty_artifact.difficulty_bin}, "
            f"source={category_artifact.source}]"
        )

        # ---- build prose chain (the model target) ------------------------
        chain = self._compose_chain(
            triplet=triplet,
            mask_artifact=mask_artifact,
            difficulty_artifact=difficulty_artifact,
            category_artifact=category_artifact,
            spatial_descriptor=spatial,
        )

        return ReasoningArtifact(
            triplet_id=triplet.triplet_id,
            header=header,
            chain=chain,
            template_version=self.config.chain_template_version,
            category=category_artifact.category,
            difficulty_bin=difficulty_artifact.difficulty_bin,
            spatial_descriptor=spatial,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _compose_chain(
        self,
        triplet: EditTriplet,
        mask_artifact: MaskArtifact,
        difficulty_artifact: DifficultyArtifactV2,
        category_artifact: CategoryArtifact,
        spatial_descriptor: str,
    ) -> str:
        """Compose the 6-step prose reasoning chain.

        Each step is one or two sentences. Steps grounded in the
        triplet's artifacts cite specific values; the category-level
        prior is hedged with "typically/often."
        """
        # Step 1: instruction reading (triplet-grounded)
        instr = (triplet.instruction or "").strip()
        # Truncate very long instructions to keep chains uniform in length
        # — chains over ~200 words risk teaching the VLM to generate padding.
        if len(instr) > 200:
            instr = instr[:197] + "..."
        step1 = (
            f"1. The edit instruction states: \"{instr}\"."
            if instr
            else "1. No explicit instruction text is available; the edit "
                 "must be inferred from visual evidence alone."
        )

        # Step 2: spatial localization (mask-grounded)
        spatial_prose = SPATIAL_DESCRIPTOR_PROSE[spatial_descriptor]
        area_pct = round(100 * mask_artifact.mask_area_frac)
        step2 = (
            f"2. The mask of changed pixels covers roughly {area_pct}% of "
            f"the image and {spatial_prose}."
        )

        # Step 3: magnitude assessment (Stage C grounded)
        struct = difficulty_artifact.structural_change
        compact_score = difficulty_artifact.compactness_score
        # Compactness inversion: compactness_score is (1 - compactness),
        # so low compact_score means the mask region is well-concentrated.
        compactness_phrase = (
            "well-concentrated in a single coherent region"
            if compact_score < 0.3
            else "diffuse or split across multiple sub-regions"
            if compact_score > 0.6
            else "moderately concentrated"
        )
        magnitude_phrase = (
            "minor"
            if struct < 0.25
            else "substantial"
            if struct > 0.55
            else "moderate"
        )
        step3 = (
            f"3. Structural change relative to the original is {magnitude_phrase} "
            f"(SSIM-based score = {struct:.2f}), and the edit region is "
            f"{compactness_phrase}."
        )

        # Step 4: category classification (Stage D grounded)
        cat = category_artifact.category
        conf = category_artifact.confidence
        if category_artifact.source == "dataset_label":
            cat_evidence = (
                "based on the dataset's curated edit-type label"
                if conf >= 0.99
                else "from the dataset label, though that label is not in "
                     "our canonical mapping"
            )
        elif category_artifact.source == "rule_based":
            cat_evidence = (
                f"inferred from the instruction text via a rule-based "
                f"keyword match (confidence {conf:.2f})"
            )
        else:  # fallback
            cat_evidence = (
                "could not be determined confidently from the available "
                "signals; treated as an unspecified edit type"
            )
        step4 = f"4. The edit is classified as {cat}, {cat_evidence}."

        # Step 5: forensic signature (category-level prior)
        prior = get_forensic_prior(cat)
        step5 = (
            f"5. Edits of this type typically exhibit {prior}."
        )

        # Step 6: difficulty estimate (Stage C grounded)
        bin_label = difficulty_artifact.difficulty_bin
        difficulty_raw = difficulty_artifact.difficulty_raw
        instr_complexity = difficulty_artifact.instruction_complexity
        bin_phrase = (
            "easier than average to detect, given clear local geometry "
            "and a low-complexity instruction"
            if bin_label == "easy"
            else "harder than average to detect, given diffuse geometry "
                 "or a high-complexity instruction"
            if bin_label == "hard"
            else "of moderate detection difficulty"
        )
        step6 = (
            f"6. Overall, this triplet is {bin_phrase} "
            f"(difficulty score = {difficulty_raw:.2f}, instruction "
            f"complexity = {instr_complexity:.2f})."
        )

        return "\n".join([step1, step2, step3, step4, step5, step6])

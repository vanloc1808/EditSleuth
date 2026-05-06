"""Stage D: CategoryClassifier.

Given an `EditTriplet` (and optionally its `MaskArtifact`), assign one
of the canonical `EditCategory` labels.

Two-path design
---------------
The classifier dispatches between two strategies:

1. **Dataset-label mapping**: if the triplet's metadata carries a
   per-row source category (e.g., Pico-Banana's ``source_edit_type``),
   map it through a curated lookup table to the canonical taxonomy.
   This is the high-confidence, no-inference path.

2. **Rule-based classification**: when no source label is present,
   apply rules over the instruction text plus optional mask scope to
   pick a category. Modest confidence; explicit "other" fallback when
   no rule fires confidently.

The two paths produce identical-shape `CategoryArtifact` records
distinguished only by the ``source`` field.

Why no learned classifier
-------------------------
A learned text classifier (e.g., fine-tuned BERT on labeled instructions)
would likely outperform rule-based on free-text instructions, but:

* For Pico-Banana the rule-based path is irrelevant — we use dataset
  labels.
* For MagicBrush instructions, free-text patterns are simple enough
  that rules cover the dominant cases. The forensic detector that
  trains downstream gets supervised by category; that's where
  learning-vs-rules matters, not here.
* Adding a learned classifier introduces training data dependencies
  that aren't justified for a categorization step that runs once.

If empirical category-quality on MagicBrush ever needs improvement,
swap the rule-based path for a learned one — the artifact shape and
driver are agnostic to the inference mechanism.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from edit2forensics.data.category_artifact import (
    CategoryArtifact,
    EDIT_CATEGORIES,
    EditCategory,
)
from edit2forensics.data.mask_artifact import MaskArtifact
from edit2forensics.data.triplet import EditTriplet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Pico-Banana source-label mapping
# ---------------------------------------------------------------------- #
# Pico-Banana's release uses a 35-category taxonomy. Each source string
# maps to one of our 10 canonical categories. Keys are lowercased and
# stripped during lookup, so source-label whitespace/case differences
# don't break the map.
#
# When adding new datasets that carry per-row labels (e.g., a future
# UltraEdit adapter), extend this dict — mappings live in code rather
# than YAML so changes are diffable and reviewable.
PICO_BANANA_LABEL_MAP: dict[str, EditCategory] = {
    # ============================================================
    # The label strings below are the EXACT source_edit_type values
    # found in Pico-Banana's release manifest, lowercased and stripped.
    # Built by inspecting the actual parquet output of
    # scripts/scan_source_labels.py.
    #
    # Mapping decisions follow the Pico-Banana paper's own 8-category
    # grouping where the grouping has clear forensic motivation:
    #
    # * "Pixel & Photometric" → ``photometric`` canonical category.
    #   Carves film grain and color-tone shifts out of artistic style
    #   transfer because they're thin photometric overlays vs. deep
    #   semantic transformations.
    # * "Human-Centric" → ``human_centric`` canonical category. Bundles
    #   the 14 person-specific operations (pose, clothing, accessories,
    #   plus 9 person-stylization variants like Funko-Pop / LEGO /
    #   Simpsonize / anime / Pixar / Western comic / sticker / line-art
    #   / caricature). All share a forensic signature: subject-
    #   localized, identity-preserving transformation of a human figure.
    # * "Scene Composition & Multi-Subject" → ``scene_transformation``.
    #   Includes "Add new scene context/background" (per the paper's
    #   grouping) — adding scene context is a full-scene operation,
    #   not a localized background swap.
    # * "Spatial/Layout" + "Scale" → ``geometric``. The paper splits
    #   these into one-element sub-categories (Zoom in / Outpainting);
    #   we merge them since both are spatial-canvas operations and
    #   the per-category metric value of further splitting is low.
    #
    # The remaining decisions match the paper's groupings 1:1:
    # * "Object-Level Semantic" → object_addition / object_removal /
    #   object_replacement / attribute_change / geometric (relocate,
    #   size/shape/orientation).
    # * "Stylistic" → style_transfer.
    # * "Text & Symbol" → text_edit.
    # ============================================================

    # === object_addition === (14,187)
    "add a new object to the scene": "object_addition",

    # === object_removal === (15,109)
    "remove an existing object": "object_removal",

    # === object_replacement === (14,547)
    "replace one object category with another": "object_replacement",

    # === attribute_change === (24,600 = 13,813 + 10,787)
    # Object-Level Semantic operations on non-human objects.
    "change an object's attribute (e.g., color/material)": "attribute_change",
    "change the size/shape/orientation of an object": "attribute_change",

    # === style_transfer === (42,874 = 15,283 + 14,855 + 12,734)
    # Pico-Banana paper's "Stylistic" category — full-image artistic
    # style transformations, not photometric overlays.
    "strong artistic style transfer (e.g., van gogh/anime/etc.)": "style_transfer",
    "modern ↔ historical style/look": "style_transfer",
    "photo to cartoon/sketch/comic": "style_transfer",

    # === photometric === (30,188 = 15,443 + 14,745)
    # Pico-Banana paper's "Pixel & Photometric" category — global
    # tonal/filter effects that don't change the underlying image
    # content semantics.
    "add film grain or vintage filter": "photometric",
    "change overall color tone (warm ↔ cool)": "photometric",

    # === scene_transformation === (52,692 = 13,438 + 12,432 + 11,992 + 14,830)
    # Pico-Banana paper's "Scene Composition & Multi-Subject" category.
    # All whole-scene environmental edits including scene-context
    # addition (which the paper groups here).
    "apply seasonal transformation (summer ↔ winter)": "scene_transformation",
    "adjust global lighting (golden hour/fluorescent)": "scene_transformation",
    "change weather conditions (sunny/rainy/snowy)": "scene_transformation",
    "add new scene context/background": "scene_transformation",

    # === background_change === (0 on Pico-Banana)
    # Reserved for MagicBrush-style "change the background to X"
    # operations where the foreground is preserved. Pico-Banana's
    # "scene context/background" is more expansive (full scene
    # composition) and goes to scene_transformation per the paper.

    # === text_edit === (10,688 = 3,866 + 3,494 + 1,897 + 1,431)
    "add new (handwritten/printed/etc) text": "text_edit",
    "replace text in signs/posters/billboards": "text_edit",
    "translate written text into other languages": "text_edit",
    "change font style or color of visible text if there is text": "text_edit",

    # === geometric === (32,742 = 13,729 + 12,402 + 6,611)
    # Pico-Banana paper's "Scale" + "Spatial/Layout" + relocation.
    "zoom in": "geometric",
    "outpainting (extend canvas beyond boundaries)": "geometric",
    "relocate an object (change its position/spatial relation)": "geometric",

    # === human_centric === (19,930 total)
    # Pico-Banana paper's "Human-Centric" category — 14 operations
    # all on a person subject. Identity-preserving transformations.
    # Person-specific attribute edits (5,907):
    "clothing edit (change color/outfit)": "human_centric",
    "change age / gender": "human_centric",
    "add/remove/replace accessories (glasses, hats, jewelry, masks)": "human_centric",
    "modify expressions (smile, frown, neutral)": "human_centric",
    "pose tweak (minor plausible change)": "human_centric",
    # Person-stylization variants (14,023):
    "funko-pop–style toy figure of the person": "human_centric",
    "lego-minifigure rendition of the person": "human_centric",
    "line-art ink sketch of the person": "human_centric",
    "simpsonize the person (yellow-skin cartoon style)": "human_centric",
    "sticker-ify the person with bold outline and white border": "human_centric",
    "convert person to 2d anime/manga style (identity-preserving)": "human_centric",
    "convert person to pixar/disney-like 3d cartoon look": "human_centric",
    "convert person to western comic cel-shaded style": "human_centric",
    "caricature with mild feature exaggeration (keep identity)": "human_centric",
}


# ---------------------------------------------------------------------- #
# Rule-based classification (for label-less datasets, e.g. MagicBrush)
# ---------------------------------------------------------------------- #
# Each rule is (compiled_pattern, category, confidence, rationale_tag).
# Rules are evaluated in order; the first match wins. Order matters
# only when patterns overlap — e.g. "replace the cup" matches both
# "replace" (replacement) and "cup" (object), so the replacement rule
# is listed first to take precedence.

@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern
    category: EditCategory
    confidence: float
    rationale: str


# NOTE: this rule set is a STARTING POINT. A pass over a MagicBrush
# instruction sample with category labels would let us refine
# precision/recall per rule. For the immediate paper-deadline scope,
# these rules cover the dominant patterns; we revisit if the empirical
# distribution looks off.
_RULES: list[_Rule] = [
    # ============================================================
    # Style transfer (high precedence — strong domain keywords).
    # Comes first because instructions like "convert to watercolor"
    # would otherwise be caught by the broad turn-into rule.
    # ============================================================
    _Rule(re.compile(r"\b(style of|in the style|artistic|van gogh|monet|picasso)\b", re.I),
          "style_transfer", 0.85, "style-of"),
    _Rule(re.compile(r"\b(cartoon|comic|sketch|painting|anime|watercolor|oil paint)\b", re.I),
          "style_transfer", 0.80, "style-keyword"),

    # ============================================================
    # Scene transformation (precedes turn-into and add — "add some
    # snow" is a weather change, not an object addition).
    # MagicBrush uses inflected forms ("raining", "snowing"); we
    # match the stem with a permissive suffix to catch those.
    # ============================================================
    _Rule(re.compile(r"\b(weather|rain(?:ing|y)?|snow(?:ing|y)?|fog(?:gy)?|"
                     r"cloud(?:y|s)?|sunny|storm(?:y|ing)?|wind(?:y)?)\b", re.I),
          "scene_transformation", 0.70, "weather"),
    _Rule(re.compile(r"\b(season|winter|summer|autumn|fall|spring)\b", re.I),
          "scene_transformation", 0.70, "season"),
    # Holiday / seasonal scene words. MagicBrush has compact phrasings
    # like "change to christmas" — the holiday name is the only signal.
    _Rule(re.compile(r"\b(christmas|halloween|easter|thanksgiving|"
                     r"hanukkah|diwali|valentine|new year)\b", re.I),
          "scene_transformation", 0.75, "holiday"),
    _Rule(re.compile(r"\b(night(?:time)?|day(?:time)?|morning|evening|"
                     r"sunset|sunrise|dusk|dawn|midnight|noon)\b", re.I),
          "scene_transformation", 0.70, "time-of-day"),
    _Rule(re.compile(r"\b(lighting|illuminat|shadow|dark(?:er|en)?|bright(?:er|en)?)\b", re.I),
          "scene_transformation", 0.65, "lighting"),

    # ============================================================
    # Object removal (precedes background_change because "erase
    # the background text" should fire as removal — the operative
    # verb is "erase").
    # ============================================================
    _Rule(re.compile(r"\b(remove|delete|erase|get rid of|take off|take out)\b", re.I),
          "object_removal", 0.85, "remove"),

    # ============================================================
    # Background change (precedes object_replacement; "change the
    # background to a sunset" should be background_change, not
    # turn-into-replacement).
    # ============================================================
    _Rule(re.compile(r"\b(background|backdrop)\b", re.I),
          "background_change", 0.75, "background"),

    # ============================================================
    # Object replacement (explicit verbs only — "replace", "swap",
    # "turn X into Y").
    # ============================================================
    _Rule(re.compile(r"\breplace\b", re.I), "object_replacement", 0.85, "replace"),
    _Rule(re.compile(r"\bswap\b", re.I), "object_replacement", 0.80, "swap"),
    _Rule(re.compile(r"\bturn\b\s+(?:\w+\s+){1,3}into\b", re.I),
          "object_replacement", 0.70, "turn-into"),

    # ============================================================
    # Object addition.
    # MagicBrush expansion: "give him X" / "have X wear Y" /
    # "let X have Y" / "X holding/wearing Y" all describe adding
    # something. We catch these via the implied-addition verbs.
    # ============================================================
    _Rule(re.compile(r"\b(add|insert|put|place)\b", re.I),
          "object_addition", 0.80, "add"),
    _Rule(re.compile(r"\bgive\b\s+(?:him|her|it|them|the\s+\w+)\b", re.I),
          "object_addition", 0.75, "give"),
    _Rule(re.compile(r"\b(wear(?:ing|s)?|hold(?:ing|s)?)\b", re.I),
          "object_addition", 0.65, "wear-hold"),

    # ============================================================
    # Text edit (specific keywords; precedes attribute_change
    # because "change the text" matches both).
    # ============================================================
    _Rule(re.compile(r"\b(text|sign|label|caption|word|letter)\b", re.I),
          "text_edit", 0.70, "text-keyword"),

    # ============================================================
    # Attribute change.
    # MagicBrush-friendly expansion: any color keyword in a
    # transformational frame ("let X be red", "make it blue", "have
    # X be green"), not just the explicit verb forms.
    # ============================================================
    _Rule(re.compile(r"\b(make|change|recolor|repaint).*\b(color|red|blue|green|yellow|"
                     r"black|white|gold|silver|purple|orange|pink|brown|gray|grey)\b", re.I),
          "attribute_change", 0.80, "color"),
    # Conversational frames + color word (catches "let it be red",
    # "have him be orange", "the cat is now red"). Lower confidence
    # because the frame is permissive.
    _Rule(re.compile(r"\b(let|have|make|is now|are now)\b\s+(?:\w+\s+){0,3}\b"
                     r"(red|blue|green|yellow|black|white|gold|silver|"
                     r"purple|orange|pink|brown|gray|grey)\b", re.I),
          "attribute_change", 0.70, "frame-color"),
    # Size / shape comparative adjectives. Catches "the dog is bigger",
    # "make it taller", etc. Comparatives (-er) are the most common
    # MagicBrush form; we also include explicit superlatives.
    _Rule(re.compile(r"\b(bigger|smaller|larger|taller|shorter|longer|wider|"
                     r"thinner|thicker|fatter|skinnier|huge|tiny|massive)\b", re.I),
          "attribute_change", 0.70, "size"),
    _Rule(re.compile(r"\b(material|texture|wooden|metal|plastic|glass|leather|fabric)\b", re.I),
          "attribute_change", 0.75, "material"),

    # ============================================================
    # Geometric transformations.
    # ============================================================
    _Rule(re.compile(r"\b(crop|zoom|rotate|flip|mirror)\b", re.I),
          "geometric", 0.85, "geometric-op"),
]


# ---------------------------------------------------------------------- #
# Classifier
# ---------------------------------------------------------------------- #


class CategoryClassifier:
    """Assign canonical edit categories to triplets.

    Stateless. Instantiate once per process. The ``classify()`` method
    dispatches to dataset-label mapping when a source label is
    available, falling back to rule-based instruction analysis
    otherwise.
    """

    def __init__(
        self,
        label_maps: dict[str, dict[str, EditCategory]] | None = None,
        rules: list[_Rule] | None = None,
        unknown_label_confidence: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        label_maps
            Per-source-dataset label maps. Keys are dataset names
            matching ``EditTriplet.source_dataset``; values are
            ``{source_label_lower → canonical_category}`` dicts.
            Defaults to ``{"pico_banana": PICO_BANANA_LABEL_MAP}``.
        rules
            Ordered list of rules for the rule-based path. Defaults
            to the module-level ``_RULES``. Override for custom
            heuristics or unit testing.
        unknown_label_confidence
            Confidence assigned when a source label exists but isn't
            in the map (treated as ``other``, with a rationale that
            includes the unknown label). Default ``0.5`` reflects
            "we know there is some structured category we just can't
            map" — higher than the no-rule-fired fallback (which gets
            the rule-based path's lower confidence).
        """
        self.label_maps = label_maps or {"pico_banana": PICO_BANANA_LABEL_MAP}
        self.rules = list(rules) if rules is not None else list(_RULES)
        self.unknown_label_confidence = unknown_label_confidence

    def classify(
        self,
        triplet: EditTriplet,
        mask_artifact: MaskArtifact | None = None,
    ) -> CategoryArtifact:
        """Return a `CategoryArtifact` for this triplet.

        Parameters
        ----------
        triplet
            The triplet to classify. Pulls from ``triplet.metadata``
            for source labels; falls back to ``triplet.instruction``
            for rule-based paths.
        mask_artifact
            Optional Stage B output. Currently unused by the rule
            base, but reserved for future rules that need mask scope
            (e.g., a "global edit but no style keyword" → infer
            scene_transformation rule).
        """
        # Path 1: dataset label
        source_label = self._extract_source_label(triplet)
        if source_label is not None:
            return self._classify_from_label(triplet, source_label)

        # Path 2: rule-based on instruction
        return self._classify_from_rules(triplet)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _extract_source_label(self, triplet: EditTriplet) -> str | None:
        """Pull a per-row source category label from triplet metadata
        if one is available.

        For Pico-Banana, this is ``metadata['source_edit_type']``.
        Other adapters may use other keys; extend this method as
        adapters grow.

        Note on the source_dataset key: the Pico-Banana adapter writes
        ``"pico-banana"`` (with hyphen). We normalize by replacing
        hyphens with underscores so the dispatch is robust to future
        adapter naming choices ("pico_banana", "pico-banana", etc.
        all match).
        """
        normalized = (triplet.source_dataset or "").replace("-", "_").lower()
        if normalized == "pico_banana":
            label = (triplet.metadata or {}).get("source_edit_type")
            if isinstance(label, str) and label.strip():
                return label
        return None

    def _classify_from_label(
        self, triplet: EditTriplet, source_label: str,
    ) -> CategoryArtifact:
        label_norm = source_label.strip().lower()
        # Normalize source_dataset the same way _extract_source_label does
        # so the label_map lookup is robust to hyphen/underscore differences
        # in adapter naming.
        dataset_key = (triplet.source_dataset or "").replace("-", "_").lower()
        label_map = self.label_maps.get(dataset_key, {})
        canonical = label_map.get(label_norm)
        if canonical is not None:
            return CategoryArtifact(
                triplet_id=triplet.triplet_id,
                category=canonical,
                confidence=1.0,
                source="dataset_label",
                rationale=f"label:{source_label}",
            )
        # Source label exists but isn't mapped — known limitation.
        return CategoryArtifact(
            triplet_id=triplet.triplet_id,
            category="other",
            confidence=self.unknown_label_confidence,
            source="dataset_label",
            rationale=f"unmapped:{source_label[:80]}",
        )

    def _classify_from_rules(self, triplet: EditTriplet) -> CategoryArtifact:
        text = triplet.instruction or ""
        for rule in self.rules:
            if rule.pattern.search(text):
                return CategoryArtifact(
                    triplet_id=triplet.triplet_id,
                    category=rule.category,
                    confidence=rule.confidence,
                    source="rule_based",
                    rationale=f"rule:{rule.rationale}",
                )
        # No rule matched — fallback.
        return CategoryArtifact(
            triplet_id=triplet.triplet_id,
            category="other",
            confidence=0.2,
            source="fallback",
            rationale=f"no-rule-matched:{text[:60]}",
        )


def category_distribution(artifacts: list[CategoryArtifact]) -> dict[EditCategory, int]:
    """Compute per-category counts. Always returns counts for every
    canonical category (including zeros) so downstream tooling can
    iterate the categories deterministically without missing-key
    errors."""
    out: dict[EditCategory, int] = {c: 0 for c in EDIT_CATEGORIES}
    for a in artifacts:
        out[a.category] += 1
    return out

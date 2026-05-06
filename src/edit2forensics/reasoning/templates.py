"""Per-category forensic signature templates for reasoning chains.

Each canonical edit category has a hand-curated description of the
forensic signature typical of edits in that category — what artifacts
a forensic detector should look for. These are *category-level priors*,
not per-triplet observations: the chain phrases them as "edits of this
type typically exhibit X," not "this triplet shows X."

Template versioning
-------------------
The module-level ``TEMPLATE_VERSION`` constant identifies the current
template iteration. Bump it on edits that materially change chain
content. The artifact records the version so downstream training can
filter by it if mixing chains from different versions becomes a concern.

Curation principles
-------------------
Each template is one or two sentences. Goals:

1. **Specific to the category**, not generic. "Look for boundary
   discontinuity" applies to additions, not photometric overlays.
2. **Honest about uncertainty**. Phrasing avoids absolute claims
   ("the image will show X") in favor of probabilistic ones ("often
   exhibits X" / "can exhibit X").
3. **Forensically useful**. The signatures named are real things a
   detector might use — boundary discontinuities, texture statistics,
   global histogram shifts — not unfalsifiable hand-waving.

The templates draw on standard forensic-detection literature without
citing specific papers; the goal is for the VLM to internalize the
priors, not to attribute them.
"""
from __future__ import annotations

from edit2forensics.data.category_artifact import EditCategory

TEMPLATE_VERSION = "v1.0"


# Per-category forensic signature priors. Each value is a one-or-two
# sentence statement of what a detector typically looks for in this
# category of edit. Phrased as a category-level fact, not a per-triplet
# observation — the chain is responsible for prefacing with "typically"
# or "often."
CATEGORY_FORENSIC_PRIORS: dict[EditCategory, str] = {
    "object_addition": (
        "boundary discontinuities at the edge of the inserted region, "
        "and lighting or shadow inconsistencies between the new object "
        "and its surrounding scene"
    ),
    "object_removal": (
        "inpainting artifacts where the removed object used to be, "
        "such as blurred or repeated texture patches that disagree with "
        "the surrounding context"
    ),
    "object_replacement": (
        "boundary mismatches at the silhouette of the new object, plus "
        "scale or perspective inconsistencies if the replacement does "
        "not match the original object's geometry"
    ),
    "attribute_change": (
        "color or texture discontinuities along the object's boundary "
        "where the edited region meets its preserved surroundings, often "
        "without changes elsewhere in the image"
    ),
    "style_transfer": (
        "global texture and brushstroke patterns inconsistent with "
        "natural photography, applied uniformly across the image regardless "
        "of original content"
    ),
    "photometric": (
        "a global histogram shift or noise overlay applied uniformly to "
        "all pixels, with the underlying image content semantically "
        "unchanged from the original"
    ),
    "scene_transformation": (
        "globally consistent changes in lighting, color temperature, or "
        "weather effects that affect the whole scene coherently rather "
        "than any single object"
    ),
    "background_change": (
        "a sharp transition between a preserved foreground subject and a "
        "newly-introduced background, sometimes with mismatched lighting "
        "or perspective at the boundary"
    ),
    "text_edit": (
        "font or rendering artifacts in the modified text region — "
        "inconsistent letter spacing, mismatched typefaces, or rendering "
        "noise distinct from the original photographic text"
    ),
    "geometric": (
        "canvas-level transformations such as cropped boundaries, scaled "
        "content, or extrapolated regions outside the original frame, "
        "rather than localized object edits"
    ),
    "human_centric": (
        "subject-localized stylization or attribute changes confined to a "
        "person, with the surrounding scene preserved; identity-preserving "
        "transformations often introduce distinctive rendering artifacts "
        "around the face and hair"
    ),
    "other": (
        "edit characteristics depend on the specific operation; without a "
        "confirmed category, look broadly for any local boundary "
        "discontinuities or global statistical shifts"
    ),
}


def get_forensic_prior(category: EditCategory) -> str:
    """Return the forensic-signature prior for a category.

    Defensive: if the category somehow isn't in the map (e.g., a typo
    in some upstream code), falls back to the ``other`` template
    rather than raising — this is a description string, not control
    flow, so degrading gracefully is preferable to crashing the
    annotator on a single bad row.
    """
    return CATEGORY_FORENSIC_PRIORS.get(
        category, CATEGORY_FORENSIC_PRIORS["other"]
    )

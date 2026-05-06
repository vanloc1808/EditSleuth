"""Stage C: DifficultyScorer.

Computes a per-triplet difficulty score from the four components
prescribed by the design doc:

* Structural change   — ``1 - SSIM(real, edited)``
* Perceptual change   — mean of Stage B's combined diff map
* Locality            — ``1 - mask_area_frac`` (smaller edits = harder)
* Instruction complexity — pluggable rule- or LLM-based score over the
                            edit instruction text.

The scorer is split into:

* ``InstructionComplexityScorer`` — pluggable strategy with a default
  rule-based implementation. Add LLM-based variants without touching
  the orchestrator.
* ``DifficultyScorer`` — combines all four components with a configurable
  weight vector and produces a ``DifficultyArtifact`` per triplet.

Tertile binning is *not* done here per-row — it requires the full
distribution. The driver script does it as a second pass after all
rows are scored.
"""

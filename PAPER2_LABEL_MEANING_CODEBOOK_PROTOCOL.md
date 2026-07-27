# Paper 2 diagnostic: label surface versus defined meaning

Status: frozen before GPU output, 2026-07-24.

## Motivation

The output-label effect survives exact readout-position matching. That result
rules out question semantics alone and neutral token-count differences, but it
does not distinguish arbitrary surface-token identity from learned semantic
meaning.

This experiment independently crosses an arbitrary terminal label with an
explicitly defined response meaning while holding the question, latent field,
readout position, worlds, and intervention fixed.

## Frozen design

Model, 30 compatible worlds, narrative surface, original belief question,
Alice-cube belief field (`ac`), L21 source interchange, cumulative L22-L27
full-readout-attention clamps, and all existing gates remain unchanged.

Surface labels:

1. `ALPHA`;
2. `BETA`;
3. `GAMMA`.

Defined meanings:

1. `BELIEF`;
2. `TELL`;
3. `SEARCH`.

Each prompt states: "In this response code, [surface] denotes a [meaning]
response." It then requests the same belief location and requires exactly
`[surface] [location]`. Neutral `X` padding is inserted after the state marker
using a deterministic tokenizer-only procedure so all nine readout positions
are identical:

1. enumerate zero through 16 padding tokens for each frozen cell;
2. choose the smallest readout position shared by all nine cells;
3. use the smallest padding count reaching that position;
4. abort with `POSITION_MATCH_INVALID` if no shared position exists.

Padding selection observes tokenization only, never model outputs.

Primary outcomes at cumulative L24:

- mean minimum mediation by defined meaning, averaged over surface labels;
- mean minimum mediation by surface label, averaged over meanings.

Secondary outcome: first passing cumulative prefix in every cell.

## Frozen verdicts

Let `meaning_range` and `surface_range` be max-minus-min of the corresponding
three L24 means.

- `DEFINED_MEANING_DOMINANT`: all cells are eligible and source sufficient,
  positions match, `meaning_range >= 0.05`, and
  `meaning_range >= 2 * surface_range`.
- `SURFACE_TOKEN_DOMINANT`: positions and gates pass,
  `surface_range >= 0.05`, and `surface_range >= 2 * meaning_range`.
- `MIXED_CODE_FACTORS`: positions and gates pass and either range is at least
  0.05, but neither dominance rule holds.
- `NO_CODE_FACTOR_EFFECT`: both ranges are below 0.05.
- `POSITION_MATCH_INVALID`: a common position cannot be constructed or final
  readout positions differ.
- `BEHAVIORALLY_INELIGIBLE`: any baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: any cell fails to pass through L27.

A defined-meaning result would show that response semantics dynamically
condition causal routing despite a fixed arbitrary output token. A
surface-token result would instead expose a lexical readout-circuit effect.


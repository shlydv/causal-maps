# Paper 2 diagnostic: Qwen-14B replication behavior screen

Status: frozen before GPU output, 2026-07-24.

## Purpose

The first Qwen-14B route run was formally ineligible because original SEARCH
natural behavior was 46.7%. Diagnose that failure without viewing any new
mechanistic intervention and define a fresh behaviorally eligible bucket for
one locked rerun.

## Frozen design

Use the same Qwen-2.5-14B-Instruct-AWQ model, 30 tokenizer-compatible worlds,
belief question, `ac` field, and position-matched `BELIEF` / `X X SEARCH`
contracts.

- Diagnostic split: compatible-world positions 15-29, exactly the first
  replication rows.
- Candidate split: positions 0-14, not used in the first Qwen-14B replication.
- Run only untouched BELIEF and SEARCH baselines on clean and natural arms.
- Save each world's predicted location, correctness, and gold-versus-best-other
  location-logit margin.
- A candidate world is eligible only if all four cells are correct:
  BELIEF clean, BELIEF natural, SEARCH clean, and SEARCH natural.
- Preserve candidate order and select every eligible world. Do not rank by
  margin or inspect route effects.
- Require at least 8 eligible candidate worlds to authorize a rerun.

The screen contains no activation transplant, route clamp, or source
intervention. If at least 8 worlds qualify, their exact indices and this
screen artifact hash will be frozen into a separate rerun protocol before
another GPU output is observed. Thresholds and prompts remain unchanged.

## Outcomes

- `ELIGIBLE_BUCKET_AVAILABLE`: at least 8 candidate worlds pass.
- `ELIGIBLE_BUCKET_TOO_SMALL`: fewer than 8 pass.
- `TOKEN_ALIGNMENT_INVALID`: the original position-matching invariants fail.

This diagnostic cannot itself support the candidate mechanism.

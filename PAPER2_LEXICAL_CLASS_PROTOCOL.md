# Paper 2 diagnostic: pretrained lexical-class generalization

Status: frozen before GPU output, 2026-07-24.

## Motivation

Exact position matching preserved the original BELIEF/TELL/SEARCH label
effect, while arbitrary ALPHA/BETA/GAMMA labels did not inherit the effect
from an in-context semantic definition. This nominates pretrained lexical
representations. The present experiment tests whether the routing profile
generalizes to unseen synonymous labels or is peculiar to the three original
tokens.

## Frozen design

Model, 30 compatible worlds, narrative surface, original belief question,
Alice-cube belief field (`ac`), L21 source interchange, cumulative L22-L27
full-readout-attention clamps, and all existing gates remain unchanged.

Three frozen lexical classes are tested:

- epistemic: `BELIEF`, `THINK`, `KNOW`;
- communication: `TELL`, `REPORT`, `SAY`;
- search/action: `SEARCH`, `FIND`, `LOOK`.

Neutral leading `X` tokens position-match all nine output contracts with a
tokenizer-only procedure:

1. enumerate zero through 16 leading `X` tokens for each base label;
2. choose the smallest readout position shared by all nine labels;
3. use the smallest padding count reaching that position;
4. emit `POSITION_MATCH_INVALID` if no common position exists.

Padding selection observes tokenization only, never model outputs. The
question, expected field, worlds, and readout position are identical.

Primary outcome: mean cumulative-L24 minimum mediation by lexical class.
Secondary outcome: first passing cumulative prefix for each label.

The original anchor words are `BELIEF`, `TELL`, and `SEARCH`; the remaining
six words form a held-out synonym check.

## Frozen verdicts

- `LEXICAL_CLASS_GENERALIZATION`: all cells are eligible and source
  sufficient; positions match; full class means are strictly
  epistemic > communication > search/action with epistemic-minus-search at
  least 0.05; and the held-out synonym means preserve
  epistemic > search/action by at least 0.03.
- `ANCHOR_WORD_ONLY_EFFECT`: gates and positions pass, the full
  epistemic-minus-search difference is at least 0.05, but the held-out
  synonym criterion fails.
- `OTHER_LEXICAL_STRUCTURE`: gates and positions pass and the range across
  full class means is at least 0.05, but neither result above applies.
- `NO_LEXICAL_CLASS_EFFECT`: the full class-mean range is below 0.05.
- `POSITION_MATCH_INVALID`: a common position cannot be constructed or final
  positions differ.
- `BEHAVIORALLY_INELIGIBLE`: any baseline fails.
- `SOURCE_SITE_INELIGIBLE`: baselines pass but any L21 source intervention
  fails.
- `DEPTH_UNRESOLVED`: any cell fails to pass through L27.

A positive result supports a pretrained semantic lexical class that
conditions causal routing. It does not yet identify the internal label
representation responsible for the modulation.


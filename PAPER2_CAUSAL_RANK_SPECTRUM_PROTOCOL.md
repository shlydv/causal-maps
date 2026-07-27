# Paper 2 causal-rank spectrum protocol

Status: frozen before model execution  
Date: 2026-07-26  
Stage: `delta_causal_rank_spectrum`

## Question

Is the validated color residual `R` one universal switch, one axis of a small
structured control space, or a task-specific leftover?

This experiment does not assume that a causal geometry exists. It measures the
causal dimensionality of a space constructed only from natural command
contrasts.

## Donor-only basis

At the three answer-prefix positions after layer 21, construct eleven mean
natural BELIEF-minus-SEARCH controllers:

- all nine pairings of `BELIEF/THINK/KNOW` with `SEARCH/FIND/LOOK` on 15
  location donor worlds;
- one ownership controller from 15 donor histories;
- one color controller from 15 donor histories.

No causal result or evaluation activation enters the basis. Apply uncentered
SVD to the flattened `3 x hidden_size` controllers. This creates eleven
orthogonal axes ordered by natural representational energy.

## Held-out causal targets

- Location: the remaining 15 compatible worlds.
- Color: 30 histories absent from all basis-construction histories, using the
  previously frozen irrelevant carrier sentence.

Ownership is not a target because its frozen behavioral gate was ineligible.

## Frozen interventions

For each target:

- untouched BELIEF and SEARCH;
- exact within-world natural answer-prefix interchange;
- cumulative reconstruction using ranks `1, 2, 3, 4, 6, 11`;
- three norm-matched random directions.

For color only:

- axes `1` through `6` separately;
- axes `7--11` as one frozen tail group.

Every intervention uses the coefficient of that target's donor controller on
the frozen basis. Evaluation rows never determine direction, scale, rank or
sign.

## Measurements

- Bidirectional movement of the established layer-24 route score.
- Fraction of the rank-11 reconstruction recovered at each cumulative rank.
- Number and identity of independently active color axes.
- Per-world bidirectional sign consistency.
- Clean behavioral eligibility and source-intervention sufficiency.
- Natural energy carried by each axis versus its causal route effect.

## Frozen gates

- Rank-11 reconstruction and natural interchange must each have bidirectional
  score at least `0.03` on both targets.
- The causal rank is the smallest cumulative rank recovering at least `80%`
  of the rank-11 score, moving both directions correctly, and showing both
  predicted signs in at least `80%` of worlds.
- A color axis is independently active when its bidirectional score is at least
  `0.015`, at least `20%` of the rank-11 score, both aggregate directions are
  correct, and at least `80%` of worlds have both predicted signs.

## Interpretation

- `SINGLE_SHARED_CAUSAL_AXIS`: rank one explains both targets and no second
  color component is independently active.
- `LOW_RANK_STRUCTURED_CAUSAL_SUBSPACE`: both targets need at most four axes
  and color contains at least two independent causal components or an active
  low-energy tail.
- `HIGH_RANK_OR_DOMAIN_SPECIFIC_CONTROL`: color requires more than four axes or
  cannot recover the full controller coherently.
- Eligibility/reference failures and unresolved patterns have separate
  verdicts.

A low-rank structured pass would justify a larger behavior-level test and the
claim that `R` is one axis of a small causal control space. A single-axis pass
would support a universal-switch hypothesis. A high-rank result would close
the general causal-space story and reclassify `R` as task-specific.

# Prospective three-mode latent route-selector protocol

Status: frozen before model evaluation.

## Question

The binary compiler assay could not distinguish a destination-addressed
instruction from generic eviction: with only two modes, moving away from the
source necessarily means moving toward the sole alternative. This assay asks
the discriminating question:

> Can the identity of a training-derived early code select one of two
> alternative destinations from the same source, and can the same destination
> code select that destination from either other source?

The tested modes are the established position-matched pretrained lexical
route classes:

- epistemic: `BELIEF / THINK / KNOW`;
- communication: `TELL / REPORT / SAY`;
- search: `SEARCH / FIND / LOOK`.

This is a test of latent route selection, not yet a claim that arbitrary
natural-language instructions or output computations can be replaced.

## Prospective construction

For each candidate early layer and destination mode \(t\), fit one six-state
destination code from training data only:

\[
z_t =
\mathbb{E}_{x,h}
\left[
h_t(x,h)-\frac{1}{2}
\sum_{s\ne t}h_s(x,h)
\right].
\]

The expectation covers four training computation families, two lexical
panels, disjoint training worlds and both clean/natural histories. There is
no fitted scale. The same \(z_t\) is used when starting from either of the two
other modes.

Candidate layers are `L2, L6, L10, L14, L18`. Selection uses only two
calibration families and the `THINK / REPORT / FIND` panel. It maximizes the
minimum held-in destination-specific score over all twelve
family/source/target cells. Earlier layer is the deterministic tie-breaker.
The selected layer, three codes, random controls, layouts and test split are
hashed before target evaluation.

Held-out evaluation uses:

- unseen `KNOW / SAY / LOOK` labels;
- unseen directed world pairs;
- unseen maximum-score and two-hop-pointer computation families;
- all six directed source-to-target transitions in each family;
- twelve rows and two histories per cell (`24` row interventions);
- twelve family-direction cells in total.

## Primary target-selection measurements

For a source \(s\), intended target \(t\), and the remaining decoy \(d\):

1. target progress measures movement from the native source state toward the
   native same-row target state at generated L21 and processed L27;
2. destination accuracy asks whether the patched state is closer to the
   native target than to both the source and decoy states;
3. target-versus-decoy accuracy asks whether it is closer to the target than
   the decoy even if the source remains nearest;
4. code-identity margin subtracts progress toward \(t\) under code \(z_d\)
   from progress toward \(t\) under \(z_t\).

The same target code must pass from both possible non-target starts. Merely
leaving the source, or reaching either non-source mode, cannot pass.

## Frozen controls

Every test cell contains:

1. exact same-row target state at L21;
2. exact same-row target-minus-source state at the selected early layer;
3. the training-derived destination code at all six command positions;
4. the competing destination code from the same source;
5. sign reversal of the intended destination code;
6. the intended code at six position-matched identical-token positions;
7. nineteen seeded isotropic random codes, each exactly norm-matched to the
   intended code;
8. early destination code plus restoration of the native source answer state
   at L21;
9. transplantation of the code-generated L21 answer state without the early
   intervention;
10. receiver-specific generated-state matching against nineteen frozen row
    derangements.

Because all nineteen random codes have identical norm to their corresponding
learned destination code, effect-size variation within that null population
cannot be attributed to norm.

## Frozen gates

Calibration proceeds to held-out testing only when the selected layer has
minimum value accuracy at least `80%` and minimum destination-specific score
at least `0.10`.

The exact early capacity oracle passes a cell when:

- mean generated-L21 progress is at least `0.60`;
- mean processed-L27 progress is at least `0.40`;
- at least `20/24` rows move positively at both checkpoints;
- minimum value accuracy is at least `80%`.

A destination code passes a cell when:

- mean generated-L21 and processed-L27 progress are each at least `0.20`;
- it recovers at least `35%` of the exact-early L27 effect;
- at least `20/24` rows move positively at both checkpoints;
- target-versus-decoy accuracy is at least `20/24` at both checkpoints;
- full three-mode destination accuracy is at least `18/24` at L27;
- its L27 progress toward the intended target exceeds the competing
  destination code by at least `0.10`;
- minimum value accuracy is at least `80%`.

Causal mediation requires L21 source-state restoration to remove at least
`70%` of the primary L27 effect and generated-state rescue to recover at
least `70%`.

Specificity requires:

- every one of the twelve destination-code cells to pass;
- all three destinations to pass from both alternative sources in both test
  families;
- sign-reversal breadth and identical-position breadth no greater than half
  the primary breadth;
- no frozen norm-matched random code to equal or exceed the primary
  twelve-cell breadth (add-one empirical `p = 0.05`);
- receiver-matched residual reconstruction to beat all nineteen frozen row
  derangements with at least `10%` median error reduction in every cell.

## Verdicts

- `DESTINATION_SPECIFIC_CONTEXTUAL_SELECTOR`: every exact, destination-code,
  source-independence, mediation, contextualization and specificity gate
  passes.
- `SELECTIVE_STEERING_WITHOUT_NATIVE_MEDIATION`: all destination-code cells
  pass, but mediation, contextual reconstruction or global null specificity
  does not.
- `GENERIC_EVICTION_NOT_SELECTION`: early codes move away from their sources
  broadly, but code identity does not select the preregistered destination.
- `NO_PORTABLE_THREE_MODE_CONTROL`: exact early targets work broadly but the
  frozen destination codes do not move all routes reliably.
- `NO_EARLY_THREE_MODE_CAPACITY`: late targets are interpretable but exact
  early target states do not broadly generate the later target routes.
- `CAUSAL_TARGET_UNRESOLVED`: exact matched L21 targets fail broadly.
- `CALIBRATION_SELECTOR_NULL`: the prospective selector fails calibration,
  so held-out target activations are never evaluated.

No prompt, mode, family, lexical panel, split, layer, scale, site, threshold,
random seed or verdict rescue follows the output.

# Addressed-state reasoning ladder (discovery)

Version 2026-07-14-v1. This is a Qwen-only discovery kernel, not confirmation.
Each prompt defines three disjoint two-transition chains using nine single-token
values. CLEAN starts on chain one; NATURAL changes only the start value to
chain two. The answer is the endpoint after exactly two transitions.

G0 requires >=80% CLEAN and NATURAL next-token accuracy. If eligible, inject
at fixed L2 and the start-value address either (a) the raw neutral-carrier
target-minus-source state, (b) the corresponding raw embedding difference, or
(c) a neutral-carrier wrong-chain direction. No scaling or layer search is
allowed. Consequence propagation requires >=80% target accuracy and positive
rows, ADD/NATURAL ratio in [.70,1.30], and a margin over the wrong chain.

For discovery only, capture the final query state at frozen layers
4/8/12/16/20/26. At each layer, exchange the matched NATURAL and CLEAN query
states in both directions. Report the earliest layer at which NATURAL->CLEAN
matches the natural effect and target accuracy while CLEAN->NATURAL restores
the source answer. This localizes a candidate causal workspace for a later
cross-fitted intervention; the matched patch is not itself evidence for a
reusable latent controller.

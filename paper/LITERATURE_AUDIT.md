# Literature audit (cutoff: 12 July 2026)

## Verdict

**Publishable as a short methodological note, with a narrower mechanistic claim than the internal frozen note.**

The vector construction itself is not novel. Difference-of-means activation directions, activation addition, task vectors, and function vectors were established before this project. Cross-prompt steering and its failures have also been studied directly. The publishable result is the controlled contrast:

> A pre-registered template-replication gate rejects the full layer-by-position activation-patching map, while a direction extracted at the causal site transfers across held-out surfaces at approximately within-template strength.

This is an empirical counterexample to the inference “non-replicating site-map implies no reusable intervention.” It is not evidence that activation directions in general are new, nor sufficient evidence that the model’s natural computation literally “lives” in the direction.

## Closest prior work

1. **Activation patching and localization.** Meng et al. (2022), Wang et al. (2023), Heimersheim and Nanda (2024), and Zhang and Nanda (2024) establish causal tracing/patching and show that corruption and metric choices can change localization. Our result changes the scored object: full site-map replication fails although a held-out direction-transfer test passes.
2. **Task/function vectors.** Hendel et al. (2023) and Todd et al. (2024) show compact activation vectors that causally drive in-context tasks and transfer beyond their extraction contexts. Nadaf (2026) runs 4,032 directed cross-template function-vector tests, making cross-template direction transfer itself clearly non-novel. These papers preclude any claim that “directions are the discovery.”
3. **Activation addition.** Turner et al. (2023) and Rimsky et al. (2024) establish contrastive mean-difference directions and residual-stream addition. Our method is a controlled application of that established intervention.
4. **Generalization and reliability.** Tan et al. (2024) find that steering vectors can generalize across prompts but are uneven and brittle; Braun et al. (2025) connect reliability to directional agreement. Méloux et al. (2025) analyze high variance in mechanistic scores under resampling and paraphrase. Our result adds a paired site-map/direction comparison under the same frozen micro-skill protocol.
5. **Prompt-specific or distributed mechanisms.** Franco et al. (2026) find prompt-specific circuit structure; Bayat Makou et al. (2026) find structurally different but functionally interchangeable circuits; Cheng and Zhang (2026) find tasks for which single-position intervention fails but coordinated multi-position intervention succeeds. These results block any broad claim that directions dominate sites or that site fragility is universal.
6. **Causality versus invariance.** Opiełka et al. (ICLR 2026) show that causally effective function vectors can be format-specific, while different concept vectors generalize across formats. This is complementary and sets a hard boundary: our “position-free” result covers a 13-token inert-prefix shift, not arbitrary format or language invariance.
7. **Cross-environment standards.** Long (2025) proposes necessity, sufficiency, and invariance across predicate-preserving references as an acceptance rule. Our template and position transfers supply sufficiency plus limited invariance, but not a full circuit-level necessity test.
8. **Intervention faithfulness.** Makelov et al. (2024) show that subspace interventions can activate dormant pathways; Grant et al. (ICLR 2026 Oral) show that causal interventions can create divergent representations. Our random-direction, embedding, wrong-value, and anti-direction controls establish specificity relative to those controls, but do not measure latent-distribution divergence or rule out a dormant pathway.

## Claim audit

### Supported

- Site-map template replication is not necessary for a transferable activation-space intervention.
- At Qwen2.5-7B, held-out Variable and Completion surfaces accept donor directions at approximately within-template strength.
- For Variable, the same protocol replicates separately at Qwen2.5-1.5B and transfers in both directions across a 13-token position shift at 7B.
- The 7B Variable direction is layer-computed relative to the embedding control and signed relative to anti-direction addition.
- The wrong-value direction also transfers, so the isolated operation is coarse: “update the value slot,” not “bind this lexical value.”

### Not supported

- That directions, task vectors, or activation addition are novel.
- That both skills were tested at both model scales (Completion was tested at 7B; Variable at 1.5B and 7B).
- That a direction was transferred between model scales; the residual spaces differ and vectors were extracted separately.
- That the direction is invariant across arbitrary formats, languages, or model families.
- That successful ADD identifies the exact natural mechanism or rules out intervention-induced/dormant pathways.
- That a value-specific binder has been isolated.

## Publication framing

Use **“transferable intervention direction”** rather than **“the causal content lives in a direction.”** The latter is stronger than the current faithfulness controls support after Makelov et al. and Grant et al.

Recommended title:

> **Fragile Patching Maps Can Hide Reusable Causal Directions**

Recommended one-sentence claim:

> In two controlled micro-skills at 7B, full layer-by-position patching maps fail a frozen template-replication gate, yet directions extracted at their causal sites transfer across held-out surfaces at approximately within-template strength; the Variable protocol separately replicates at 1.5B and transfers across a 13-token position shift at 7B, while a wrong-value control identifies the direction as a coarse slot update rather than a value-specific binder.

## Research implication

The result motivates scoring both **where** an intervention works and **what direction** transfers. The next pre-registered experiment should separate the generic wrong-value subspace from the matched direction and test the orthogonal residual for value-selective transfer. Composition is a later horizon, not evidence in this note.

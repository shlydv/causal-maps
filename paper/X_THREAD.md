# X thread draft

Voice check: lowercase, direct, no em dashes, no inflated novelty claim. Replace `[paper link]` and `[repo link]` only after the release pages exist.

## Thread

**1/10**

i spent the last month asking a simple mechanistic question:

if an activation patching map does not replicate across prompt templates, does that mean the mechanism itself is not reusable?

in two small controlled tasks, the answer was no.

[paper link]

attach: `figures/coordinate_system.png`

---

**2/10**

first, the negative result.

at qwen2.5-7b, layer x token maps failed my preregistered full-grid, template-disjoint gate:

variable substitution: rho 0.37  
completion state: rho 0.14  
required: rho > 0.5 plus a calibrated null

both failed.

---

**3/10**

but variable substitution had a strange structure.

the full maps were fragile across templates, with mean pairwise rho 0.26.

the causal column at the value token was almost identical across templates, r = 0.999.

the site effect was stable. the rest of the map was not.

---

**4/10**

so i tested a different object.

instead of asking whether the same site map reappears, i took the mean counterfactual minus clean residual direction at the causal site and added it to held-out templates.

cross-template effect / within-template effect = 1.01.

---

**5/10**

this was not just one variable prompt.

the same test transferred:

completion surfaces at 7b: ratio 1.00, 100% greedy flips  
variable at 1.5b, extracted separately: ratio 1.02  
position 27 to 40: ratio 1.09  
position 40 to 27: ratio 0.92

attach: `figures/transfer_ratios.png`

---

**6/10**

controls narrowed the claim.

an embedding direction was basically inert, only 0.005 of the layer-2 effect.

adding the negative direction hurt.

so the effect is signed and computed inside the network, not a token embedding copy.

---

**7/10**

the important caveat:

an unrelated wrong-value direction transferred at ratio 0.59. cosine with matched was 0.77.

this triggered my preregistered generic_boost verdict. it is not value-specific.

it looks more like a coarse “update this value slot” operation.

attach: `figures/controls.png`

---

**8/10**

directions are not new. task vectors, function vectors, and activation addition are established.

the result here is the mismatch between two evaluation objects:

the site map says “unreplicable.”  
the held-out direction test says “reusable.”

those are different claims.

---

**9/10**

recent work makes the faithfulness limit clear.

causal vectors need not be format-invariant. activation interventions can also leave the natural latent distribution or recruit dormant pathways.

so i claim a transferable intervention, not the model’s unique natural circuit.

---

**10/10**

the takeaway:

a fragile patching heatmap does not imply that no reusable causal intervention exists.

if a study uses site-map replication to judge mechanism reuse, it should test direction transfer separately.

paper: [paper link]  
code + artifacts: [repo link]

## Single-post version

activation-patching maps for two qwen2.5-7b micro-skills failed template replication, but residual directions at their causal sites transferred across held-out surfaces at within-template strength. variable also transferred at 1.5b and across a 13-token position shift. a wrong-value control still transferred, so the direction is a coarse slot update, not a value-specific binder. result: fragile site maps can hide reusable interventions. [paper link]

## Suggested release order

1. Upload the arXiv version and create a stable repository release.
2. Post 1/10 with the coordinate-system figure.
3. Add the transfer-ratio figure to 5/10.
4. Add the control figure to 7/10.
5. Keep 8/10 and 9/10 in the main thread. They are not optional caveats.

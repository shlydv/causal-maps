# Cross-model binding-operator gate

Version: 2026-07-14-v1. This is a limited replication gate, not a claim of
shared internal circuitry.

For each model, retain only the fixed ten candidate value words that are a
single mid-sentence tokenizer token. The run is ineligible if fewer than eight
remain. Construct the standard two-binding substitution trials. Within each
source value, offsets 1/3 are discovery and offsets 5/7 are held out.

The only injection-depth candidates are normalized early layers
`round((L - 1) * {0.06, 0.12, 0.20})`, clipped to valid non-final layers.
At each depth, donor value prototypes are the mean residual state from the
fixed names X/Y/Z/W at that layer. Add `prototype(target)-prototype(source)`
at the queried binding's value token.

Discovery selection requires behavioral accuracy >= .70, a positive natural
and ADD effect, >= .70 positive ADD rows, ADD/natural in [.50, 1.50], and ADD
greater than a wrong-direction ADD by .10 logit-difference units. Select the
viable depth with largest ADD effect.

Held-out confirmation is stricter: clean and natural answer accuracy >= .80,
positive natural and ADD effects, >= .80 positive ADD rows, ADD/natural in
[.70, 1.30], ADD > wrong-direction by .10, and ADD > same delta at the other
binding slot by .10. A result confirms only a cross-model *behavioral affine
operator* at a selected early binding slot. It does not establish an identical
controller, circuit, or distributed mechanism.

The planned family replication order is Mistral-7B-Instruct-v0.3 followed by
Phi-3.5-mini-instruct. Phi is selected over a gated Gemma checkpoint because
the experiment must be reproducible in a fresh Kaggle environment without a
personal access token; its official model card specifies `trust_remote_code`.

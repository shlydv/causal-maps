# Cross-surface binding-operator replication

Version: 2026-07-14-v2. This tests whether the affine binding-operator
phenomenon recurs under an out-of-distribution surface form, not whether one
literal residual vector transfers across grammars.

The only new prompt family is:

`In the table, X maps to VALUE_X; Y maps to VALUE_Y. What does X map to?`

The assistant is primed with `X =` or `Y =`, according to the queried binding.
The v1 free-form completion `It maps to` was declared behaviorally ineligible
on Qwen: it makes an article the next token, so it cannot support a
single-next-token value test. Values are dynamically restricted to
the fixed candidate words that are single tokens for each model. The gate is
ineligible below eight values. Use only offsets 5/7 from the fixed mapping
construction; there is no prompt, value, layer, or threshold search.

For each model, form L2 value prototypes from separate mapping-surface donor
prompts with names X/Y/Z/W. At the queried value token add
`prototype(target)-prototype(source)`. Measure NATURAL, ADD, wrong-direction
ADD at the correct slot, and target ADD at the other binding slot.

Confirmation requires CLEAN and NATURAL answer accuracy >= .80, positive
NATURAL/ADD effects, >= .80 positive ADD rows, ADD/NATURAL in [.70,1.30], and
ADD at least .10 larger than both controls. The only planned models are
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3. A pass demonstrates a
recurrent surface-independent *phenomenon*; it does not show that the literal
direction or full circuit is shared across grammars.

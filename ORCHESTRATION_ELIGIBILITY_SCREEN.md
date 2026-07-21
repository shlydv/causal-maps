# Screen: Is a structured tool workflow behaviorally eligible?

*Causal Maps · 2026-07-13*
*Status: **COMPLETE — `ORCHESTRATION_ELIGIBLE`.***

## Workflow

Each prompt contains:

- two payload numbers;
- one database key;
- two available tools;
- one mode token: `calculate` or `lookup`.

Required fixed-width calls:

- `CALL calculator <first> <second>`
- `CALL database <key> 0`

The harness executes the call:

- calculator returns the sum;
- database returns the key's hidden mapped digit.

The generated call and tool result are then placed into a follow-up conversation,
where the model must return the result as its final answer.

## Frozen setup

- Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, 8-bit,
  bitsandbytes 0.49.2, seed 0.
- 20 payload rows.
- Exact mode pairs must have equal token length and one changed token.
- Tool calls are parsed from the entire normalized pre-EOS continuation;
  explanations or extra lines fail exactness.
- Greedy generation only; maximum 8 action tokens and 4 answer tokens.
- Save every generated call and final response.

## Gate

For both calculate and lookup:

- exact tool-call accuracy ≥90%;
- correct tool name and arguments ≥90%;
- final-answer accuracy after tool execution ≥90%.
- end-to-end accuracy (correct action and correct final answer on the same row)
  ≥90%.

Verdicts:

- `ORCHESTRATION_ELIGIBLE` if both modes pass every threshold;
- `ORCHESTRATION_INELIGIBLE` otherwise.

No activation is extracted in this screen. An eligible result unlocks one
separate preregistered causal orchestration kernel.

## Result

Both modes scored 100% on every frozen metric:

- exact calls;
- correct tools and row-specific arguments;
- final answers after actual parsed-call execution;
- same-row end-to-end success.

Verdict: **`ORCHESTRATION_ELIGIBLE`**. The separate causal orchestration kernel
is unlocked.

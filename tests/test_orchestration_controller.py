"""Offline regression tests for orchestration parsing and verdict logic."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_orchestration_controller import (  # noqa: E402
    _safe_ratio, _verdict)
from causal_maps.delta_orchestration_screen import (  # noqa: E402
    _correct_action, _execute, _normalize, _parse_call, _rows)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


row = _rows()[0]
calculator = _parse_call(f"CALL calculator {row['a']} {row['b']}")
database = _parse_call(f"CALL database {row['key']} 0")
check(_correct_action(row, "calculate", calculator), "calculator action")
check(_correct_action(row, "lookup", database), "database action")
check(not _correct_action(row, "lookup", calculator), "wrong-mode action")
check(_execute(calculator) == str(row["a"] + row["b"]), "calculator execution")
check(_execute(database) == str(row["database_value"]), "database execution")
check(_execute(_parse_call("CALL calculator A 1")) is None, "malformed calculator")
check(_execute(_parse_call("CALL database 1 0")) is None, "malformed database")
check(
    _parse_call(_normalize("CALL database A 0\nextra<|im_end|>")) is None,
    "extra output must fail")

base = {
    "G0": True, "A1": True, "O1": True, "W1": True, "R1": True,
    "Q1": True, "M1": True, "M2": True, "B1": True,
}
check(_verdict(base) == "LATENT_ORCHESTRATION_CONTROLLER", "latent verdict")
check(
    _verdict({**base, "B1": False}) == "LEXICAL_ORCHESTRATION_REPLAY",
    "lexical verdict")
check(
    _verdict({**base, "Q1": False}) == "ORCHESTRATION_ALTERNATE_PATH",
    "alternate verdict")
check(
    _verdict({**base, "A1": False})
    == "ORCHESTRATION_OPERATOR_AMBIGUOUS",
    "ambiguous verdict")
check(
    _verdict({**base, "O1": False}) == "ORCHESTRATION_CONTROL_NULL",
    "null verdict")

check(_safe_ratio(1.0, 0.0) is None, "invalid ratio")
strict = json.dumps(
    {"ratio": _safe_ratio(1.0, 0.0)}, allow_nan=False)
check(strict == '{"ratio": null}', "strict JSON")
check(not any(
    isinstance(value, float) and not math.isfinite(value)
    for value in json.loads(strict).values() if value is not None),
    "finite JSON values")

print("orchestration controller regression tests passed")

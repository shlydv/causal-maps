"""causal_maps: activation-patching harness for causal maps of micro-skills.

All heavy imports (torch, transformers) live in submodules so that importing
the package for pure-logic use (pair generators, nulls) stays light.
"""
__all__ = ["logutil", "model_utils", "patching", "nulls", "rule_world", "binding_pairs"]

"""Offline contract checks for the fixed reversed-layout early replication."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_backup_reversed_confirmation import (  # noqa: E402
    EARLY_LAYERS, LATE_READ_LAYERS)


def main():
    assert EARLY_LAYERS == (3, 4, 5, 6, 7, 8)
    assert LATE_READ_LAYERS == (21, 22, 23, 24, 25, 26)
    print("binding backup reversed confirmation contract tests passed")


if __name__ == "__main__":
    main()

import numpy as np

from causal_maps.delta_entity_matrix import _make_rows
from causal_maps.delta_preprint_battery import _full_depth_layers
from causal_maps.delta_preprint_probe import _balanced_rows
from causal_maps.delta_structured_workspace import LOCATIONS, _rows
from causal_maps.delta_workspace_matrix import _make_rows as make_matrix_rows


def test_workspace_widening_has_30_distinct_transitions_per_cell():
    for i, name in enumerate(("retrieve", "add2", "sub1", "max5", "gt5label")):
        rows = make_matrix_rows(name, np.random.default_rng(i), 30)
        assert len(rows) == 30
        assert len({(a, b) for a, b, _ in rows}) == 30


def test_entity_widening_has_unique_worlds():
    vals = ["red", "blue", "green", "black", "white", "brown", "pink", "gray"]
    rows = _make_rows("keys", vals, None, np.random.default_rng(0), 30)
    keys = [tuple(row[k] for k in ("a", "b", "w", "oa", "ob", "ow"))
            for row in rows]
    assert len(keys) == len(set(keys)) == 30


def test_anchor_census_exhausts_distinct_nuisance_pairs():
    rows = _rows("Paris", "Rome", "ac", "test", n_rows=30)
    assert len(rows) == 30
    assert len({(row["tc"], row["ts"]) for row in rows}) == 30


def test_full_depth_grid_reaches_late_14b_layers():
    layers = _full_depth_layers(48)
    assert 46 in layers
    assert max(layers) >= 46
    assert all(0 <= layer < 48 for layer in layers)


def test_probe_rows_are_label_balanced_and_truth_is_noncolliding():
    rows, labels, reps = _balanced_rows(n_reps=6, seed=2718)
    assert len(rows) == 48
    for rep in range(6):
        assert sorted(labels[reps == rep].tolist()) == list(range(len(LOCATIONS)))
    assert all(row["tc"] != row["ts"] for row in rows)

import numpy as np

from causal_maps.delta_entity_matrix import _make_rows
from causal_maps.delta_paper1_closeout import (NAT_SOURCE, NAT_TARGET, STYLES,
                                               _naturalized_rows)
from causal_maps.delta_preprint_battery import _full_depth_layers
from causal_maps.delta_preprint_locus import (_build_loci, _curve_verdict)
from causal_maps.delta_preprint_probe import _balanced_rows
from causal_maps.delta_structured_workspace import LOCATIONS, _rows
from causal_maps.delta_workspace_matrix import _make_rows as make_matrix_rows
from causal_maps.delta_verbalization import _reverse_base_verdict


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


def _reverse_panel(hist_rome, hist_paris, verbal_rome, verbal_paris,
                   both_paris=1.0, both_lam=1.0):
    return {
        "g0": [1.0, 1.0],
        "history_reverse": {"rome_acc": hist_rome,
                            "paris_acc": hist_paris},
        "verbal_reverse": {"rome_acc": verbal_rome,
                           "paris_acc": verbal_paris},
        "both_reverse": {"paris_acc": both_paris, "lam": both_lam},
    }


def test_reverse_base_verdict_distinguishes_quorum_from_prior():
    quorum = _reverse_panel(1.0, 0.0, 1.0, 0.0)
    prior = _reverse_panel(0.0, 1.0, 0.0, 1.0)
    assert _reverse_base_verdict(quorum) == "QUORUM_REPLICATES_REVERSE_BASE"
    assert (_reverse_base_verdict(prior) ==
            "PARIS_PRIOR_REPLICATES_REVERSE_BASE")


def test_reverse_base_verdict_protects_sanity_and_mixed_branches():
    sanity = _reverse_panel(1.0, 0.0, 1.0, 0.0, both_paris=0.7)
    mixed = _reverse_panel(1.0, 0.0, 0.0, 1.0)
    assert _reverse_base_verdict(sanity) == "REVERSE_BASE_SANITY_FAIL"
    assert _reverse_base_verdict(mixed) == "REVERSE_BASE_MIXED"


def test_locus_sets_are_nested_disjoint_and_leave_one_out():
    anchors = {"ac": 10, "bc": 20, "as": 30,
               "bs": 40, "tc": 50, "ts": 60}
    summary = [90, 91, 92, 93, 94, 95]
    loci = _build_loci(95, summary, anchors, sequence_length=120,
                        n_random=3, seed=2718)
    union = set(anchors.values()) | set(summary)
    assert set(loci["marker_only"]) <= set(loci["summary_span"])
    assert set(loci["anchors_plus_summary"]) == union
    assert set(loci["anchors_plus_summary"]) <= set(loci["full_prequery"])
    assert set(loci["full_prequery"]) < set(loci["full_matched_prefix"])
    for field, position in anchors.items():
        loo = set(loci[f"anchors_without_{field}"])
        assert position not in loo
        assert loo == set(anchors.values()) - {position}
    for i in range(3):
        random_locus = set(loci[f"random_size_matched_{i}"])
        assert len(random_locus) == len(union)
        assert random_locus.isdisjoint(union)


def test_locus_verdict_requires_full_prefix_upper_bound():
    def cell(ok=False):
        return {"forward_ratio": 1.0 if ok else 0.0,
                "reverse_ratio": 1.0 if ok else 0.0,
                "forward_target_acc": 1.0 if ok else 0.0,
                "reverse_clean_acc": 1.0 if ok else 0.0}

    names = ("marker_only", "summary_span", "source_anchors",
             "anchors_plus_summary", "full_prequery",
             "full_matched_prefix")
    curve = {8: {name: cell(False) for name in names}}
    assert _curve_verdict(curve) == "UPPER_BOUND_FAILED"
    curve[8]["full_matched_prefix"] = cell(True)
    assert _curve_verdict(curve) == "QUERY_OR_READOUT_CONTEXT_REQUIRED"
    curve[8]["full_prequery"] = cell(True)
    assert _curve_verdict(curve) == "DISTRIBUTED_PREQUERY_SUFFICIENT"
    curve[8]["anchors_plus_summary"] = cell(True)
    assert _curve_verdict(curve) == "LOCAL_UNION_SUFFICIENT"
    curve[8]["source_anchors"] = cell(True)
    assert _curve_verdict(curve) == "SOURCE_ANCHORS_SUFFICIENT"


def test_paper1_closeout_naturalized_rows_are_balanced_and_held_out():
    rows, groups = _naturalized_rows(30)
    assert len(rows) == 30
    assert set(groups) == set(STYLES)
    assert all(len(groups[style]) == 10 for style in STYLES)
    assert len({(row["tc"], row["ts"]) for row in rows}) == 30
    assert all(row["ac"] == NAT_SOURCE for row in rows)
    assert all(any(row[key] == NAT_TARGET for key in ("as", "bc", "bs"))
               for row in rows)

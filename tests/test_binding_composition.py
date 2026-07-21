from causal_maps.delta_binding_composition import _composition_rows, _confirmed


def test_composition_rows_are_distinct_and_query_balanced():
    rows = _composition_rows(["a", "b", "c", "d", "e", "f", "g", "h"])
    assert len(rows) == 16
    assert sum(row["query"] == "X" for row in rows) == 8
    assert all(len({row["source_x"], row["source_y"], row["source_z"],
                    row["target_x"], row["target_y"]}) == 5 for row in rows)


def test_confirmation_rejects_crosstalk():
    metrics = {"clean_acc": 1., "natural_acc": 1., "natural_effect": 10.,
               "own_only_effect": 9.7, "joint_effect": 9.6,
               "swapped_joint_effect": 3., "mean_joint_minus_own_only": -.1,
               "positive_joint_fraction": 1., "joint_to_natural_ratio": .96,
               "joint_to_own_only_ratio": .99}
    assert _confirmed(metrics)
    metrics["mean_joint_minus_own_only"] = 1.6
    assert not _confirmed(metrics)

import numpy as np
import torch

from causal_maps.delta_entity_matrix import _make_rows
from causal_maps.delta_paper1_closeout import (NAT_SOURCE, NAT_TARGET, STYLES,
                                               _naturalized_rows)
from causal_maps.delta_preprint_battery import _full_depth_layers
from causal_maps.delta_preprint_locus import (_build_loci, _curve_verdict)
from causal_maps.delta_preprint_probe import _balanced_rows
from causal_maps.delta_structured_workspace import LOCATIONS, _rows
from causal_maps.delta_shared_component import (
    _adjudicate,
    _build_component_loci,
    _component_stats,
    _fixed_split,
)
from causal_maps.delta_binding_divergence import (
    _direct_rows,
    _verdict as _binding_divergence_verdict,
)
from causal_maps.delta_content_aliasing import (
    _collision_rows,
    _pattern_verdict as _aliasing_verdict,
)
from causal_maps.delta_sparse_transport import (
    _fixed_split as _transport_split,
    _verdict as _transport_verdict,
)
from causal_maps.delta_sparse_transport_confirmation import (
    CANDIDATE_LAYERS,
    _confirmation_verdict,
    _tail_probability,
)
from causal_maps.delta_source_head_mediation import (
    _mediation_fraction,
    _mediation_pass,
    _verdict as _mediation_verdict,
)
from causal_maps.delta_conditional_backup import (
    _candidate_sites,
    _verdict as _backup_verdict,
)
from causal_maps.delta_operation_handoff_depth import (
    _first_passing,
    _verdict as _depth_verdict,
)
from causal_maps.delta_depth_replication import (
    REPLICATION_TASKS,
    _verdict as _replication_verdict,
)
from causal_maps.delta_semantic_command_factor import (
    ALL_TASKS,
    COMMAND_TASKS,
    SEMANTIC_TASKS,
    _verdict as _factor_verdict,
)
from causal_maps.delta_prompt_factorial import (
    ALL_TASKS as FACTORIAL_TASKS,
    COMMANDS as FACTORIAL_COMMANDS,
    QUESTION_NAMES as FACTORIAL_QUESTIONS,
    TASK_GRID as FACTORIAL_GRID,
    _verdict as _factorial_verdict,
)
from causal_maps.delta_token_length_ladder import (
    ALL_TASKS as LENGTH_TASKS,
    FILLER_COUNTS,
    QUESTION_NAMES as LENGTH_QUESTIONS,
    TASK_GRID as LENGTH_GRID,
    _verdict as _length_verdict,
)
from causal_maps.delta_position_matched_label import (
    ALL_TASKS as MATCHED_TASKS,
    LABEL_NAMES as MATCHED_LABELS,
    QUESTION_NAMES as MATCHED_QUESTIONS,
    TASK_GRID as MATCHED_GRID,
    _verdict as _matched_verdict,
)
from causal_maps.delta_label_meaning_codebook import (
    ALL_TASKS as CODE_TASKS,
    MEANINGS as CODE_MEANINGS,
    SURFACES as CODE_SURFACES,
    TASK_GRID as CODE_GRID,
    _verdict as _codebook_verdict,
)
from causal_maps.delta_lexical_class import (
    ALL_TASKS as LEXICAL_TASKS,
    CLASSES as LEXICAL_CLASSES,
    LABELS as LEXICAL_LABELS,
    TASK_GRID as LEXICAL_GRID,
    _verdict as _lexical_verdict,
)
from causal_maps.delta_label_route_switch import (
    _verdict as _route_switch_verdict,
)
from causal_maps.delta_paired_route_transplant import (
    _verdict as _paired_transplant_verdict,
)
from causal_maps.delta_distributed_label_transplant import (
    _verdict as _distributed_transplant_verdict,
)
from causal_maps.delta_content_cancelled_controller import (
    _movement as _content_cancelled_movement,
    _norm_matched_directions,
    _verdict as _content_cancelled_verdict,
    _world_mediation,
    _world_movements,
)
from causal_maps.delta_cross_domain_controller import (
    DOMAIN_SPECS,
    _domain_rows,
    _overall_verdict as _cross_domain_overall_verdict,
    _second_largest,
)
from causal_maps.delta_controller_matrix import (
    _adjudicate as _controller_matrix_adjudicate,
    _fresh_domain_rows,
)
from causal_maps.delta_endogenous_residual_necessity import (
    _equalization_patch_pair,
    _gap_reduction,
    _nested_max,
)
from causal_maps.delta_controller_circuit_epistasis import (
    _adjudicate as _controller_circuit_adjudicate,
    _discover_gate_heads,
    _epistasis_metrics,
    _fresh_color_rows_v4,
    _random_head_sets,
)
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


def test_paper2_component_split_is_disjoint_and_mechanical():
    rows = [{"row": i} for i in range(30)]
    donor, evaluation = _fixed_split(rows, n_donor=15)
    assert [row["row"] for row in donor] == list(range(15))
    assert [row["row"] for row in evaluation] == list(range(15, 30))
    assert not ({row["row"] for row in donor}
                & {row["row"] for row in evaluation})


def test_paper2_component_loci_have_matched_random_sizes():
    anchors = {"ac": 10, "bc": 20, "as": 30,
               "bs": 40, "tc": 50, "ts": 60}
    summary = [90, 91, 92, 93]
    loci = _build_component_loci(
        marker=93, summary=summary, anchors=anchors, seed=7319)
    excluded = set(anchors.values()) | set(summary)
    assert loci["edited_anchor"] == [10]
    assert len(loci["random_single"]) == 1
    assert len(loci["random_belief_size"]) == len(loci["belief_anchors"])
    assert len(loci["random_source_size"]) == len(loci["source_anchors"])
    assert len(loci["random_summary_size"]) == len(loci["summary_span"])
    for name, positions in loci.items():
        if name.startswith("random_"):
            assert set(positions).isdisjoint(excluded)


def test_paper2_component_stats_detect_rank_one_alignment():
    base = np.arange(1, 13, dtype=np.float32).reshape(1, 3, 4)
    rows = np.repeat(base, 6, axis=0)
    stats = _component_stats(
        __import__("torch").from_numpy(rows))
    assert stats["mean_energy_fraction"] > .999
    assert stats["mean_pairwise_cosine"] > .999
    assert stats["top_k_energy_fraction"]["1"] > .999


def test_paper2_component_verdict_requires_specific_bidirectionality():
    passing = {
        "2": {
            "edited_anchor": {
                "rescue_pass": True,
                "necessity_pass": True,
                "bidirectional_pass": True,
            },
            "random_single": {
                "rescue_pass": False,
                "necessity_pass": False,
                "bidirectional_pass": False,
            },
        }
    }
    assert _adjudicate(True, passing) == "SHARED_CAUSAL_COMPONENT"
    passing["2"]["random_single"]["bidirectional_pass"] = True
    assert (_adjudicate(True, passing)
            == "NONLOCAL_OR_NONSPECIFIC_COMPONENT")
    assert _adjudicate(False, passing) == "BEHAVIORALLY_INELIGIBLE"


def test_paper2_direct_binding_rows_preserve_collision_structure():
    rows = _direct_rows(30)
    assert len(rows) == 30
    assert len({(row["source"], row["target"]) for row in rows}) == 30
    assert all(row["source"] != row["target"] for row in rows)
    assert all(row["ac"] == row["bc"] == row["as"] == row["source"]
               for row in rows)
    assert all(row["bs"] == row["target"] for row in rows)


def test_paper2_direct_binding_verdict_separates_the_kill_branches():
    assert (_binding_divergence_verdict(True, True, True, True)
            == "DIRECT_BINDING_SPECIFIC")
    assert (_binding_divergence_verdict(True, True, False, True)
            == "INTERVENTION_SPECIFIC_DIVERGENCE")
    assert (_binding_divergence_verdict(True, False, True, True)
            == "SYNTHETIC_CONTENT_WRITE_FAILED")
    assert (_binding_divergence_verdict(False, True, True, True)
            == "BEHAVIORALLY_INELIGIBLE")


def test_paper2_collision_rows_have_exact_source_multiplicity():
    for load in (1, 2, 3):
        rows = _collision_rows(load, 30)
        assert len(rows) == 30
        assert len({(row["source"], row["target"]) for row in rows}) == 30
        for row in rows:
            assert sum(row[r] == row["source"]
                       for r in ("A", "B", "C")) == load
            assert row["D"] == row["target"]
            assert row["source"] != row["target"]


def _alias_cell(address, alias, target=None):
    return {
        "address": {"accuracy": address},
        "alias": {"accuracy": alias},
        "target_accuracy": alias if target is None else target,
    }


def test_paper2_aliasing_verdict_requires_exact_collision_curve():
    blocks = {
        "1": {
            "synthetic": {
                "B": _alias_cell(1.0, 1.0, 0.0),
                "C": _alias_cell(1.0, 1.0, 0.0),
                "D": _alias_cell(1.0, 1.0, 1.0),
            },
            "natural_alias": {"B": 1.0, "C": 1.0},
        },
        "2": {
            "synthetic": {
                "B": _alias_cell(0.0, 1.0, 1.0),
                "C": _alias_cell(1.0, 1.0, 0.0),
                "D": _alias_cell(1.0, 1.0, 1.0),
            },
            "natural_alias": {"B": 1.0, "C": 1.0},
        },
        "3": {
            "synthetic": {
                "B": _alias_cell(0.0, 1.0, 1.0),
                "C": _alias_cell(0.0, 1.0, 1.0),
                "D": _alias_cell(1.0, 1.0, 1.0),
            },
            "natural_alias": {"B": 1.0, "C": 1.0},
        },
    }
    assert (_aliasing_verdict(True, True, blocks)
            == "CONTENT_EQUIVALENCE_ALIASING_BEHAVIORAL_AND_CAUSAL")
    blocks["1"]["synthetic"]["B"]["target_accuracy"] = 1.0
    blocks["1"]["synthetic"]["B"]["address"]["accuracy"] = 0.0
    assert (_aliasing_verdict(True, True, blocks)
            == "GLOBAL_TARGET_BROADCAST")


def test_paper2_sparse_transport_split_is_held_out():
    rows = [{"row": i} for i in range(30)]
    donor, evaluation = _transport_split(rows, 15)
    assert [row["row"] for row in donor] == list(range(15))
    assert [row["row"] for row in evaluation] == list(range(15, 30))


def test_paper2_sparse_transport_verdict_requires_random_specificity():
    top = {
        "1": {"sufficient": False},
        "2": {"sufficient": True},
        "4": {"sufficient": True},
        "8": {"sufficient": True},
    }
    random = {
        key: [{"sufficient": False} for _ in range(5)]
        for key in top
    }
    full = {"24": {"sufficient": True}}
    assert (_transport_verdict(True, top, random, full)
            == "SPARSE_TRANSPORT_PATH")
    random["2"][0]["sufficient"] = True
    assert (_transport_verdict(True, top, random, full)
            == "NONSPECIFIC_HEAD_SET")
    for cell in top.values():
        cell["sufficient"] = False
    assert (_transport_verdict(True, top, random, full)
            == "DISTRIBUTED_ATTENTION_TRANSPORT")
    full["24"]["sufficient"] = False
    assert (_transport_verdict(True, top, random, full)
            == "ATTENTION_OUTPUT_NOT_SUFFICIENT")


def test_paper2_sparse_transport_confirmation_is_locked_and_specific():
    assert CANDIDATE_LAYERS == (21, 22, 23, 24)
    tasks = {
        "tell_ac": {"eligible": True, "top4": {"sufficient": True}},
        "search_ac": {"eligible": True, "top4": {"sufficient": True}},
    }
    random = [{"sufficient": False} for _ in range(99)]
    p_value, exceed = _tail_probability(0.7, [0.1] * 99)
    assert p_value == 0.01
    assert exceed == 0
    assert (_confirmation_verdict(tasks, random, p_value)
            == "LOCKED_SPARSE_TRANSPORT_CONFIRMED")
    random[0]["sufficient"] = True
    assert (_confirmation_verdict(tasks, random, p_value)
            == "NONSPECIFIC_TRANSPORT")
    random[0]["sufficient"] = False
    tasks["search_ac"]["top4"]["sufficient"] = False
    assert (_confirmation_verdict(tasks, random, p_value)
            == "QUERY_LIMITED_TRANSPORT")
    tasks["search_ac"]["eligible"] = False
    assert (_confirmation_verdict(tasks, random, p_value)
            == "BEHAVIORALLY_INELIGIBLE")


def test_paper2_source_head_mediation_gate_and_verdict():
    source = {
        "forward_ratio": 0.8,
        "reverse_ratio": 0.7,
        "sufficient": True,
    }
    blocked = {"forward_ratio": 0.08, "reverse_ratio": 0.07}
    med = _mediation_pass(source, blocked, 1.0, 1.0)
    assert med["pass"]
    assert _mediation_fraction(0.8, 0.08) == 0.9
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
            "mediation": {"pass": name != "search_ac"},
        }
        for name in ("belief_ac", "tell_ac", "search_ac")
    }
    random = [{"mediation": {"pass": False}} for _ in range(39)]
    assert (_mediation_verdict(tasks, random, 0.025, True)
            == "OPERATION_SPECIFIC_HEAD_MEDIATION")
    tasks["search_ac"]["mediation"]["pass"] = True
    assert (_mediation_verdict(tasks, random, 0.025, True)
            == "SHARED_HEAD_MEDIATION")


def test_paper2_conditional_backup_excludes_frozen_sites_and_maps_queries():
    candidates = _candidate_sites(28)
    assert len(candidates) == 76
    assert all(site not in {
        (23, 11), (24, 21), (22, 1), (23, 6),
        (22, 25), (23, 4), (23, 13), (24, 27),
    } for site in candidates)
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in ("belief_ac", "tell_ac", "search_ac")
    }
    random = {
        name: [{"mediation": {"pass": False}} for _ in range(9)]
        for name in tasks
    }
    transfer = {
        donor: {
            target: {"mediation": {"pass": donor == target}}
            for target in tasks
        }
        for donor in tasks
    }
    assert (_backup_verdict(tasks, random, transfer)
            == "SPARSE_QUERY_COMPLEMENTS")
    for target in tasks:
        transfer["belief_ac"][target]["mediation"]["pass"] = True
    assert (_backup_verdict(tasks, random, transfer)
            == "SHARED_SPARSE_COMPLEMENT")


def test_paper2_operation_handoff_depth_detects_later_search():
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in ("belief_ac", "tell_ac", "search_ac")
    }
    first = {"belief_ac": 23, "tell_ac": 24, "search_ac": 26}
    curves = {}
    for query, threshold in first.items():
        curves[query] = {
            str(layer): {"mediation": {"pass": layer >= threshold}}
            for layer in (22, 23, 24, 25, 26, 27)
        }
    assert _first_passing(curves["search_ac"]) == 26
    assert (_depth_verdict(tasks, curves)
            == "OPERATION_DEPENDENT_HANDOFF_DEPTH")


def test_paper2_depth_replication_requires_strict_ordering():
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in REPLICATION_TASKS
    }
    thresholds = dict(zip(REPLICATION_TASKS, (23, 25, 27)))
    curves = {
        name: {
            str(layer): {
                "mediation": {"pass": layer >= thresholds[name]}}
            for layer in (22, 23, 24, 25, 26, 27)
        }
        for name in REPLICATION_TASKS
    }
    assert (_replication_verdict(tasks, curves)
            == "DEPTH_ORDERING_REPLICATED")
    thresholds = dict(zip(REPLICATION_TASKS, (24, 24, 27)))
    curves = {
        name: {
            str(layer): {
                "mediation": {"pass": layer >= thresholds[name]}}
            for layer in (22, 23, 24, 25, 26, 27)
        }
        for name in REPLICATION_TASKS
    }
    assert _replication_verdict(tasks, curves) == "MONOTONE_WITH_TIE"


def test_paper2_semantic_command_factor_clean_dissociation():
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in ALL_TASKS
    }
    thresholds = {
        **dict(zip(SEMANTIC_TASKS, (24, 26, 27))),
        **dict(zip(COMMAND_TASKS, (24, 24, 24))),
    }
    curves = {
        name: {
            str(layer): {
                "mediation": {"pass": layer >= thresholds[name]}}
            for layer in (22, 23, 24, 25, 26, 27)
        }
        for name in ALL_TASKS
    }
    assert (_factor_verdict(tasks, curves)
            == "SEMANTIC_OPERATION_EFFECT")


def test_paper2_prompt_factorial_detects_command_invariance():
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in FACTORIAL_TASKS
    }
    command_depth = {"BELIEF": 24, "TELL": 26, "SEARCH": 27}
    curves = {}
    for question in FACTORIAL_QUESTIONS:
        for command in FACTORIAL_COMMANDS:
            name = FACTORIAL_GRID[question][command]
            threshold = command_depth[command]
            curves[name] = {
                str(layer): {
                    "mediation": {"pass": layer >= threshold}}
                for layer in (22, 23, 24, 25, 26, 27)
            }
    assert (_factorial_verdict(tasks, curves)
            == "COMMAND_INVARIANT_ACROSS_QUESTIONS")


def test_paper2_token_length_ladder_detects_depth_substitution():
    tasks = {}
    curves = {}
    for question in LENGTH_QUESTIONS:
        for count in FILLER_COUNTS:
            name = LENGTH_GRID[question][count]
            tasks[name] = {
                "eligible": True,
                "readout_position": 200 + 2 * count,
                "source_intervention": {"sufficient": True},
            }
            threshold = 27 if count < 2 else 26
            curves[name] = {}
            for layer in (22, 23, 24, 25, 26, 27):
                minimum = 0.60 + 0.03 * count if layer == 24 else 0.90
                curves[name][str(layer)] = {
                    "mediation": {
                        "pass": layer >= threshold,
                        "minimum_fraction": minimum,
                    }}
    assert (_length_verdict(tasks, curves)
            == "TOKEN_LENGTH_DEPTH_SUBSTITUTION")


def test_paper2_position_matched_label_effect():
    tasks = {}
    curves = {}
    label_l24 = {"belief": 0.73, "tell": 0.64, "search": 0.62}
    label_depth = {"belief": 24, "tell": 26, "search": 26}
    for question in MATCHED_QUESTIONS:
        for label in MATCHED_LABELS:
            name = MATCHED_GRID[question][label]
            tasks[name] = {
                "eligible": True,
                "readout_position": 200,
                "source_intervention": {"sufficient": True},
            }
            curves[name] = {}
            for layer in (22, 23, 24, 25, 26, 27):
                curves[name][str(layer)] = {
                    "mediation": {
                        "pass": layer >= label_depth[label],
                        "minimum_fraction": label_l24[label],
                    }}
    assert (_matched_verdict(tasks, curves)
            == "POSITION_MATCHED_LABEL_EFFECT")


def test_paper2_codebook_detects_defined_meaning_dominance():
    tasks = {}
    curves = {}
    meaning_l24 = {"BELIEF": 0.72, "TELL": 0.66, "SEARCH": 0.60}
    for surface in CODE_SURFACES:
        for meaning in CODE_MEANINGS:
            name = CODE_GRID[surface][meaning]
            tasks[name] = {
                "eligible": True,
                "readout_position": 220,
                "source_intervention": {"sufficient": True},
            }
            curves[name] = {}
            for layer in (22, 23, 24, 25, 26, 27):
                curves[name][str(layer)] = {
                    "mediation": {
                        "pass": layer >= 26,
                        "minimum_fraction": meaning_l24[meaning],
                    }}
    assert (_codebook_verdict(tasks, curves)
            == "DEFINED_MEANING_DOMINANT")


def test_paper2_lexical_class_generalizes_beyond_anchor_words():
    tasks = {}
    curves = {}
    class_l24 = {
        "epistemic": 0.72,
        "communication": 0.66,
        "search": 0.60,
    }
    for lexical_class in LEXICAL_CLASSES:
        for label in LEXICAL_LABELS[lexical_class]:
            name = LEXICAL_GRID[lexical_class][label]
            tasks[name] = {
                "eligible": True,
                "readout_position": 220,
                "source_intervention": {"sufficient": True},
            }
            curves[name] = {}
            for layer in (22, 23, 24, 25, 26, 27):
                curves[name][str(layer)] = {
                    "mediation": {
                        "pass": layer >= 26,
                        "minimum_fraction": class_l24[lexical_class],
                    }}
    assert (_lexical_verdict(tasks, curves)
            == "LEXICAL_CLASS_GENERALIZATION")


def test_paper2_bidirectional_label_route_switch():
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief")
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in names
    }
    l24 = {
        "belief_original": 0.73,
        "search_original": 0.62,
        "belief_to_search": 0.66,
        "search_to_belief": 0.69,
    }
    first = {
        "belief_original": 24,
        "search_original": 26,
        "belief_to_search": 26,
        "search_to_belief": 24,
    }
    curves = {}
    for name in names:
        curves[name] = {}
        for layer in (22, 23, 24, 25, 26, 27):
            curves[name][str(layer)] = {
                "mediation": {
                    "pass": layer >= first[name],
                    "minimum_fraction": l24[name],
                }}
    assert (_route_switch_verdict(tasks, curves)
            == "BIDIRECTIONAL_ROUTE_SWITCH")


def test_paper2_bidirectional_paired_route_transplant():
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief")
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in names
    }
    l24 = {
        "belief_original": 0.73,
        "search_original": 0.62,
        "belief_to_search": 0.65,
        "search_to_belief": 0.70,
    }
    first = {
        "belief_original": 24,
        "search_original": 26,
        "belief_to_search": 26,
        "search_to_belief": 24,
    }
    curves = {}
    for name in names:
        curves[name] = {}
        for layer in (22, 23, 24, 25, 26, 27):
            curves[name][str(layer)] = {
                "mediation": {
                    "pass": layer >= first[name],
                    "minimum_fraction": l24[name],
                }}
    assert (_paired_transplant_verdict(tasks, curves)
            == "BIDIRECTIONAL_PAIRED_ROUTE_TRANSPLANT")


def test_paper2_specific_distributed_label_switch():
    arms = ("instruction", "answer_prefix", "all")
    names = ["belief_original", "search_original"] + [
        f"{arm}_{direction}"
        for arm in arms
        for direction in ("belief_to_search", "search_to_belief")
    ]
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in names
    }
    random_tasks = [{
        "eligible": True,
        "source_intervention": {"sufficient": True},
    } for _ in range(38)]
    l24 = {name: 0.67 for name in names}
    first = {name: 26 for name in names}
    l24.update({
        "belief_original": 0.74,
        "search_original": 0.62,
        "all_belief_to_search": 0.66,
        "all_search_to_belief": 0.70,
    })
    first.update({
        "belief_original": 24,
        "search_original": 26,
        "all_belief_to_search": 26,
        "all_search_to_belief": 24,
    })
    curves = {}
    for name in names:
        curves[name] = {}
        for layer in (22, 23, 24, 25, 26, 27):
            curves[name][str(layer)] = {
                "mediation": {
                    "pass": layer >= first[name],
                    "minimum_fraction": l24[name],
                }}
    random_controls = [
        {"bidirectional_score": 0.01}
        for _ in range(19)
    ]
    assert (_distributed_transplant_verdict(
        tasks, curves, random_tasks, random_controls)
        == "SPECIFIC_DISTRIBUTED_LABEL_SWITCH")


def test_content_cancelled_directions_match_each_position_norm():
    displacement = torch.tensor([
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 12.0],
        [1.0, 2.0, 2.0, 0.0],
    ])
    controls = _norm_matched_directions(
        displacement, n_random=4, seed=17)
    assert len(controls) == 4
    for control in controls:
        assert torch.allclose(
            control.norm(dim=-1),
            displacement.norm(dim=-1),
            atol=1e-5, rtol=1e-5)


def test_content_cancelled_world_mediation_and_direction():
    task = {"source_intervention": {
        "forward_effect_rows": [2.0, 4.0],
        "reverse_effect_rows": [5.0, 10.0],
    }}
    cell = {"blocked_intervention": {
        "forward_effect_rows": [0.5, 1.0],
        "reverse_effect_rows": [1.0, 2.0],
    }}
    route = _world_mediation(task, cell)
    assert np.allclose(route, [0.75, 0.75])
    movement = _world_movements(
        [0.8] * 15, [0.5] * 15,
        [0.7] * 15, [0.6] * 15)
    assert movement["all_predicted_sign"]
    assert np.allclose(movement["belief_to_search_range"], [0.1, 0.1])
    assert np.allclose(movement["search_to_belief_range"], [0.1, 0.1])


def test_content_cancelled_controller_verdict_requires_specific_uniform_switch():
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief",
    )
    tasks = {
        name: {
            "eligible": True,
            "source_intervention": {"sufficient": True},
        }
        for name in names
    }
    summaries = {
        "belief_original": {
            "l24_minimum_mediation": 0.74,
            "first_passing_prefix": 24,
        },
        "search_original": {
            "l24_minimum_mediation": 0.62,
            "first_passing_prefix": 26,
        },
        "belief_to_search": {
            "l24_minimum_mediation": 0.66,
            "first_passing_prefix": 26,
        },
        "search_to_belief": {
            "l24_minimum_mediation": 0.70,
            "first_passing_prefix": 24,
        },
    }
    movement = _content_cancelled_movement(
        {
            "belief": summaries["belief_original"],
            "search": summaries["search_original"],
        },
        summaries["belief_to_search"]["l24_minimum_mediation"],
        summaries["search_to_belief"]["l24_minimum_mediation"])
    worlds = {"all_predicted_sign": True}
    instruction = {"functional_bidirectional_score": 0.01}
    direction = {"empirical_p": 0.05}
    position = {"empirical_p": 0.05}
    assert (_content_cancelled_verdict(
        tasks, summaries, movement, worlds,
        instruction, direction, position)
        == "CONTENT_CANCELLED_PREFIX_CONTROLLER")


def test_cross_domain_rows_are_balanced_unique_and_noncolliding():
    for spec in DOMAIN_SPECS.values():
        rows = _domain_rows(spec["values"])
        assert len(rows) == 15
        assert len({
            (row["source"], row["target"]) for row in rows
        }) == 15
        assert all(
            len({
                row["source"], row["target"],
                row["d1"], row["d2"],
            }) == 4
            for row in rows)
        source_counts = {
            value: sum(row["source"] == value for row in rows)
            for value in spec["values"]
        }
        assert min(source_counts.values()) == 1
        assert max(source_counts.values()) == 2


def test_cross_domain_second_order_statistic_requires_two_domains():
    cells = {
        "a": {"functional_bidirectional_score": 0.12},
        "b": {"functional_bidirectional_score": 0.08},
        "c": {"functional_bidirectional_score": -0.01},
    }
    assert _second_largest(cells) == 0.08


def test_cross_domain_overall_verdict_distinguishes_universal_and_multi():
    passing = {
        name: {
            "original_evaluation_eligible": True,
            "functional_bidirectional_score": score,
            "verdict": "CROSS_DOMAIN_ROUTE_SWITCH",
        }
        for name, score in zip(DOMAIN_SPECS, (0.12, 0.10, 0.08))
    }
    instruction = {"generalization_score": 0.01}
    direction = {"empirical_p": 0.05}
    position = {"empirical_p": 0.05}
    assert (_cross_domain_overall_verdict(
        passing, instruction, direction, position)
        == "UNIVERSAL_CROSS_DOMAIN_ROUTE_CONTROLLER")
    passing["key_value"]["verdict"] = "NO_CROSS_DOMAIN_TRANSFER"
    assert (_cross_domain_overall_verdict(
        passing, instruction, direction, position)
        == "MULTI_DOMAIN_ROUTE_CONTROLLER")


def test_controller_matrix_fresh_rows_do_not_overlap_prior_pairs():
    for name in ("ownership", "color_state"):
        spec = DOMAIN_SPECS[name]
        prior = _domain_rows(spec["values"])
        fresh = _fresh_domain_rows(spec["values"])
        prior_pairs = {
            (row["source"], row["target"]) for row in prior
        }
        fresh_pairs = {
            (row["source"], row["target"]) for row in fresh
        }
        assert len(fresh) == len(fresh_pairs) == 30
        assert prior_pairs.isdisjoint(fresh_pairs)


def test_controller_matrix_adjudicates_reciprocal_coordinate_pattern():
    def cell(passed):
        return {
            "verdict": (
                "CROSS_DOMAIN_ROUTE_SWITCH"
                if passed else "NO_CROSS_DOMAIN_TRANSFER")
        }

    matrix = {
        "location": {
            "location": cell(True),
            "color_state": cell(False),
        },
        "color_state": {
            "location": cell(False),
            "color_state": cell(True),
        },
    }
    controls = {
        "location": {"specific": True},
        "color_state": {"specific": True},
    }
    gates = {
        "location": "ELIGIBLE",
        "color_state": "ELIGIBLE",
    }
    assert (_controller_matrix_adjudicate(
        matrix, controls, gates)
        == "DOMAIN_SPECIFIC_CONTROLLER_COORDINATES")
    matrix["location"]["color_state"] = cell(True)
    assert (_controller_matrix_adjudicate(
        matrix, controls, gates)
        == "ASYMMETRIC_CONTROLLER_COORDINATES")


def test_endogenous_equalization_preserves_midpoint_and_removes_coordinate():
    assert _nested_max([[0.1, 0.3], [0.2, 0.05]]) == 0.3
    generator = torch.Generator().manual_seed(19)
    belief = [
        torch.randn(5, 11, 16, generator=generator),
        torch.randn(5, 11, 16, generator=generator),
    ]
    search = [
        torch.randn(5, 11, 16, generator=generator),
        torch.randn(5, 11, 16, generator=generator),
    ]
    direction = torch.randn(3, 16, generator=generator)
    belief_patch, search_patch, invariants = _equalization_patch_pair(
        belief, search, [7, 8, 9], direction)
    assert invariants["pass"]
    for index in range(2):
        old_midpoint = 0.5 * (
            belief[index][:, [7, 8, 9], :]
            + search[index][:, [7, 8, 9], :])
        new_midpoint = 0.5 * (
            belief_patch[index][1] + search_patch[index][1])
        assert torch.allclose(
            old_midpoint, new_midpoint, atol=1e-5, rtol=1e-5)
        coordinate = (
            ((belief_patch[index][1] - search_patch[index][1])
             * direction).sum(dim=-1)
            / direction.square().sum(dim=-1))
        assert float(coordinate.abs().max()) <= 1e-5


def test_endogenous_gap_reduction_uses_paired_belief_search_contrast():
    def task(rows):
        return {
            "source_intervention": {
                "forward_effect_rows": [1.0] * len(rows),
                "reverse_effect_rows": [1.0] * len(rows),
            },
        }

    def cell(mediation_rows):
        return {
            "blocked_intervention": {
                "forward_effect_rows": [
                    1.0 - value for value in mediation_rows],
                "reverse_effect_rows": [
                    1.0 - value for value in mediation_rows],
            },
        }

    original_belief = [0.10 + index * 0.001 for index in range(30)]
    original_search = [0.02 + index * 0.001 for index in range(30)]
    equalized_belief = [0.065 + index * 0.001 for index in range(30)]
    equalized_search = [0.055 + index * 0.001 for index in range(30)]
    original_tasks = {
        "belief_original": task(original_belief),
        "search_original": task(original_search),
    }
    original_curves = {
        "belief_original": {"24": cell(original_belief)},
        "search_original": {"24": cell(original_search)},
    }
    equalized_tasks = {
        "belief_equalized": task(equalized_belief),
        "search_equalized": task(equalized_search),
    }
    equalized_curves = {
        "belief_equalized": {"24": cell(equalized_belief)},
        "search_equalized": {"24": cell(equalized_search)},
    }
    result = _gap_reduction(
        0.1145, 0.0345, 0.0795, 0.0695,
        original_tasks, original_curves,
        equalized_tasks, equalized_curves)
    assert np.isclose(result["original_gap"], 0.08)
    assert np.isclose(result["equalized_gap"], 0.01)
    assert np.isclose(result["gap_reduction"], 0.07)
    assert result["positive_reductions"] == 30
    assert result["aggregate_pass"]
    assert result["statistical_pass"]


def test_controller_circuit_fourth_split_is_unique_and_disjoint():
    rows = _fresh_color_rows_v4()
    assert len(rows) == 60
    assert len({
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }) == 120
    assert len(rows[:30]) == len(rows[30:]) == 30


def test_controller_circuit_activation_only_discovery_recovers_gate_heads():
    n_rows, n_heads, head_dim = 2, 6, 2

    def empty_bundle():
        return {
            name: torch.zeros(n_rows, n_heads * head_dim)
            for name in ("clean", "natural", "forward", "reverse")
        }

    bundles = {
        name: empty_bundle()
        for name in (
            "belief", "search",
            "belief_steered", "search_steered")
    }
    fractions = (0.1, 0.2, 0.95, 0.80, -0.1, 0.0)
    for head, fraction in enumerate(fractions):
        start = head * head_dim
        stop = start + head_dim
        gap = torch.tensor([1.0, 0.5]).repeat(n_rows, 1)
        bundles["search"]["forward"][:, start:stop] = gap
        bundles["belief_steered"]["forward"][:, start:stop] = (
            fraction * gap)
        bundles["search_steered"]["forward"][:, start:stop] = (
            (1.0 - fraction) * gap)
    discovery = _discover_gate_heads(
        bundles, list(range(n_heads)), head_dim, select_k=2)
    assert discovery["stable"]
    assert discovery["selected_heads"] == [2, 3]


def test_controller_circuit_epistasis_fraction_and_sign_gates():
    def route(score):
        return {
            "route_score": score,
            "world_route": [score] * 30,
            "public": {
                "eligible": True,
                "source_intervention": {"sufficient": True},
            },
        }

    result = _epistasis_metrics(
        route(0.70), route(0.50),
        route(0.60), route(0.60),
        route(0.68), route(0.52),
        route(0.62), route(0.58))
    assert np.isclose(
        result["blockade"]["bidirectional_fraction_score"], 0.8)
    assert np.isclose(
        result["rescue"]["bidirectional_fraction_score"], 0.8)
    assert result["blockade"]["pass"]
    assert result["rescue"]["pass"]


def test_controller_circuit_random_head_sets_are_unique_and_matched():
    controls = _random_head_sets(
        list(range(20)), selected_heads=(1, 3, 5, 7),
        n_random=19, seed=31)
    assert len(controls) == len(set(controls)) == 19
    assert all(len(cell) == 4 for cell in controls)
    assert all(
        not set(cell) & {1, 3, 5, 7}
        for cell in controls)


def test_controller_circuit_verdict_separates_one_sided_and_nonspecific():
    assert _controller_circuit_adjudicate(
        True, True, True, True, True, True, True
    ) == "CONTROLLER_GATES_TRANSPORT_CIRCUIT"
    assert _controller_circuit_adjudicate(
        True, True, True, True, False, True, False
    ) == "CONTROLLER_CIRCUIT_BLOCKADE_ONLY"
    assert _controller_circuit_adjudicate(
        True, True, True, True, False, False, True
    ) == "CONTROLLER_EFFECT_DISTRIBUTED_OR_NONSPECIFIC"

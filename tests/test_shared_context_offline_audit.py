import numpy as np

from causal_maps.analyze_context_geometry_shared_context import (
    CHECKPOINTS,
    FAMILIES,
    OPERATIONS,
    analyze,
)


def _write_maps(path, contextual):
    generator = np.random.default_rng(73)
    arrays = {}
    width = 32
    for checkpoint in CHECKPOINTS:
        for operation_index, operation in enumerate(OPERATIONS):
            shared = generator.normal(size=width)
            shared /= np.linalg.norm(shared)
            family_directions = generator.normal(
                size=(len(FAMILIES), width)
            )
            family_directions -= family_directions.mean(axis=0)
            family_directions /= np.linalg.norm(
                family_directions, axis=1, keepdims=True
            )
            for family_index, family in enumerate(FAMILIES):
                samples = []
                for history in range(2):
                    rows = []
                    for row in range(8):
                        value = shared.copy()
                        if contextual:
                            value += 0.8 * family_directions[family_index]
                        value += 0.01 * generator.normal(size=width)
                        rows.append(value.reshape(2, width // 2))
                    samples.append(rows)
                arrays[f"{family}_{operation}_L{checkpoint}"] = np.asarray(
                    samples, dtype=np.float16
                )
    np.savez_compressed(path, **arrays)


def test_reproducible_shared_plus_context_maps_license_design(tmp_path):
    source = tmp_path / "contextual.npz"
    _write_maps(source, contextual=True)

    result = analyze(source)

    assert all(result["gates"].values())
    assert result["verdict"] == (
        "PROSPECTIVE_INVERSE_CONTROL_DESIGN_LICENSED"
    )


def test_shared_maps_with_sample_noise_close_context_branch(tmp_path):
    source = tmp_path / "shared_only.npz"
    _write_maps(source, contextual=False)

    result = analyze(source)

    assert not result["gates"]["residual_reproducibility"]
    assert result["verdict"] == "SHARED_CONTEXT_GEOMETRY_BRANCH_CLOSED"

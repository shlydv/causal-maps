import torch

from causal_maps.delta_causal_rank_spectrum import (
    _component,
    _controller_basis,
)


def test_basis_reconstructs_all_natural_controllers():
    generator = torch.Generator().manual_seed(1207)
    controllers = {
        f"controller_{index}": torch.randn(
            3, 17, generator=generator)
        for index in range(11)
    }
    names, singular, basis, coefficients, geometry = (
        _controller_basis(controllers))
    assert len(names) == 11
    assert singular.shape == (11,)
    assert basis.shape == (11, 51)
    assert coefficients.shape == (11, 11)
    assert geometry["cumulative_energy"][-1] == 1.0

    for controller in controllers.values():
        reconstruction, metadata = _component(
            controller, basis, range(1, 12))
        assert torch.allclose(
            reconstruction, controller, atol=2e-5, rtol=2e-5)
        assert metadata["fraction_of_controller_energy"] > 0.9999


def test_component_groups_are_orthogonal_and_additive():
    generator = torch.Generator().manual_seed(1801)
    controllers = {
        f"controller_{index}": torch.randn(
            3, 13, generator=generator)
        for index in range(11)
    }
    _names, _singular, basis, _coefficients, _geometry = (
        _controller_basis(controllers))
    controller = controllers["controller_0"]
    head, _head_metadata = _component(
        controller, basis, range(1, 7))
    tail, _tail_metadata = _component(
        controller, basis, range(7, 12))
    assert abs(float(torch.dot(
        head.flatten(), tail.flatten()))) < 2e-4
    assert torch.allclose(
        head + tail, controller, atol=2e-5, rtol=2e-5)

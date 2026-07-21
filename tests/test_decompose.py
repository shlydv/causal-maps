"""Offline tests for the GENERIC_BOOST decomposition math (no model)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from causal_maps.delta_decompose import proj_residual  # noqa: E402
from causal_maps.delta_centroid import remove_direction  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def test_proj():
    torch.manual_seed(0)
    D = 64
    w1, w2, u0 = torch.randn(D), torch.randn(D), torch.randn(D)
    Q, _ = torch.linalg.qr(torch.stack([w1, w2], 1), mode="reduced")
    u = u0 - Q @ (Q.t() @ u0)                       # u ⊥ span(w1, w2)
    delta = 2.0 * w1 - 1.5 * w2 + 3.0 * u
    g, s = proj_residual(delta, [w1, w2])
    check("Δ = g + s", torch.allclose(g + s, delta, atol=1e-4))
    check("residual ⊥ w1", abs(float(s @ w1)) < 1e-3)
    check("residual ⊥ w2", abs(float(s @ w2)) < 1e-3)
    check("residual isolates the orthogonal part (s ≈ 3u)",
          torch.allclose(s, 3.0 * u, atol=1e-3))
    g2, s2 = proj_residual(g, [w1, w2])
    check("projection idempotent (proj(g) ≈ g)", torch.allclose(g2, g, atol=1e-4))
    check("residual of a pure-generic vector ≈ 0", float(s2.norm()) < 1e-3)
    # A vector fully inside span(others) has ~zero residual fraction
    inside = -0.7 * w1 + 2.1 * w2
    _, s_in = proj_residual(inside, [w1, w2])
    check("pure-generic Δ -> residual_frac ≈ 0",
          float(s_in.norm()) / float(inside.norm()) < 1e-3)


def test_selectivity():
    nV, vi = 5, 2
    md = np.array([0.1, 0.1, 1.0, 0.1, 0.1])          # target vi boosted
    sel = md[vi] - (md.sum() - md[vi]) / (nV - 1)
    check("selectivity > 0 when only target boosted", sel > 0)
    uni = np.full(5, 0.5)                               # uniform boost
    sel_u = uni[vi] - (uni.sum() - uni[vi]) / (nV - 1)
    check("selectivity ≈ 0 under uniform (generic) boost", abs(sel_u) < 1e-9)


def test_centroid():
    torch.manual_seed(1)
    D = 128
    vs = [torch.randn(D) for _ in range(5)]
    g = torch.stack(vs).mean(0)
    ghat = g / g.norm()
    for d in vs:
        dp = remove_direction(d, g)
        comp = float(d @ ghat)
        expected = (float(d.norm()) ** 2 - comp ** 2) ** 0.5
        check("d' orthogonal to centroid", abs(float(dp @ ghat)) < 1e-4)
        check("||d'|| = sqrt(||d||^2 - comp^2)", abs(float(dp.norm()) - expected) < 1e-3)
    check("remove_direction(g, g) ~ 0", float(remove_direction(g, g).norm()) < 1e-4)


def test_multislot_selectivity():
    from causal_maps.delta_multislot import _selectivity
    dl = np.array([[0.1, 2.0, 0.1, 0.1], [3.0, 0.0, 0.0, 0.0]])  # B=2, nV=4
    sel = _selectivity(dl, np.array([1, 0]))
    check("multislot selectivity >0 for boosted target (trial 0)", sel[0] > 0)
    check("multislot selectivity >0 for boosted target (trial 1)", sel[1] > 0)
    selu = _selectivity(np.array([[0.5, 0.5, 0.5, 0.5]]), np.array([2]))
    check("multislot selectivity ~0 under uniform", abs(selu[0]) < 1e-9)


def main():
    print("== proj_residual =="); test_proj()
    print("== selectivity =="); test_selectivity()
    print("== centroid removal =="); test_centroid()
    print("== multislot selectivity =="); test_multislot_selectivity()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()

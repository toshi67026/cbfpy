#!/usr/bin/env python

import numpy as np

from cbfpy.cbf import CircleCBF, GeneralCBF
from cbfpy.cbf_controller import CBFController


class TestCBFController:
    def test_single_cbf(self) -> None:
        """Single CBF, unconstrained case."""
        cbf = GeneralCBF(G=np.ones(2), h=10.0)
        controller = CBFController([cbf], P=np.eye(2))

        nominal_input = np.array([0.5, 0.5])
        status, optimal_input = controller.optimize(nominal_input)
        assert status == "optimal"
        # h=10 is far from boundary, so optimal ~= nominal
        assert np.allclose(optimal_input, nominal_input, atol=1e-6)

    def test_multiple_cbfs(self) -> None:
        """Multiple CBFs compose constraints correctly."""
        cbf1 = GeneralCBF(G=np.array([1.0, 0.0]), h=10.0)
        cbf2 = GeneralCBF(G=np.array([0.0, 1.0]), h=10.0)
        controller = CBFController([cbf1, cbf2], P=np.eye(2))

        nominal_input = np.array([1.0, 1.0])
        status, optimal_input = controller.optimize(nominal_input)
        assert status == "optimal"
        assert np.allclose(optimal_input, nominal_input, atol=1e-6)

    def test_active_constraint(self) -> None:
        """CBF constraint is active and modifies the input."""
        # Agent at boundary of circle (radius=1, center=0)
        cbf = CircleCBF(center=np.zeros(2), radius=1.0, keep_inside=True)
        cbf.calc_constraints(np.array([0.9, 0.0]))
        controller = CBFController([cbf], P=np.eye(2))

        # Nominal input pushes outward
        nominal_input = np.array([1.0, 0.0])
        status, optimal_input = controller.optimize(nominal_input)
        assert status == "optimal"
        # Should reduce the outward component
        assert optimal_input[0] < nominal_input[0]

    def test_p_setter(self) -> None:
        """P matrix can be updated."""
        cbf = GeneralCBF(G=np.ones(2), h=10.0)
        controller = CBFController([cbf], P=np.eye(2))
        new_P = np.diag([2.0, 1.0])
        controller.P = new_P
        assert np.allclose(controller.P, new_P)

    def test_cbf_list_property(self) -> None:
        """cbf_list property returns the CBF list."""
        cbf1 = GeneralCBF(G=np.ones(2), h=1.0)
        cbf2 = GeneralCBF(G=-np.ones(2), h=1.0)
        controller = CBFController([cbf1, cbf2], P=np.eye(2))
        assert len(controller.cbf_list) == 2

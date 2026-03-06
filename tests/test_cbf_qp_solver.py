#!/usr/bin/env python

import numpy as np
import pytest

from cbfpy.cbf_qp_solver import CBFNomQPSolver, CBFQPSolver


class TestCBFQPSolver:
    qp_solver = CBFQPSolver()
    P = np.eye(2)
    q = np.ones(2)

    def test_optimize_optimal1(self) -> None:
        """one inequality constraint (inactive)"""
        G_list = [np.ones(2)]
        alpha_h_list = [1.0]

        status, optimal_input = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)
        assert status == "optimal"
        # Unconstrained minimum: x = -q = [-1, -1], constraint [1,1]x<=1 is inactive
        assert np.allclose(optimal_input, np.array([-1.0, -1.0]))

    def test_optimize_optimal2(self) -> None:
        """two inequality constraints (one active)"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0, 1.0]

        status, optimal_input = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)
        assert status == "optimal"
        # Active constraint: x1+x2 = -1, by symmetry x1=x2=-0.5
        assert np.allclose(optimal_input, np.array([-0.5, -0.5]))

    def test_optimize_except(self) -> None:
        """dimension mismatch raises exception"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0]

        assert len(G_list) != len(alpha_h_list)
        with pytest.raises(Exception):
            _ = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)


class TestCBFNomQPSolver:
    nom_qp_solver = CBFNomQPSolver()
    P = np.eye(2)
    nominal_input = np.ones(2)

    def test_optimize_optimal1(self) -> None:
        """one inequality constraint (inactive)"""
        G_list = [np.ones(2)]
        alpha_h_list = [1.0]

        status, optimal_input = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)

        assert status == "optimal"
        # Unconstrained minimum: u = nominal_input = [1, 1], constraint is inactive
        assert np.allclose(optimal_input, np.array([1.0, 1.0]))

    def test_optimize_optimal2(self) -> None:
        """two inequality constraints (one active)"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0, 1.0]

        status, optimal_input = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)

        assert status == "optimal"
        # Active constraint: u1+u2 = 1, by symmetry u1=u2=0.5
        assert np.allclose(optimal_input, np.array([0.5, 0.5]))

    def test_optimize_except(self) -> None:
        """dimension mismatch raises exception"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0]

        assert len(G_list) != len(alpha_h_list)
        with pytest.raises(Exception):
            _ = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)

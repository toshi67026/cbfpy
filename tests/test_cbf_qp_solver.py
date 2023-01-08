#!/usr/bin/env python

import numpy as np
import pytest  # noqa

from cbfpy.cbf_qp_solver import CBFNomQPSolver, CBFQPSolver


class TestCBFQPSolver:
    qp_solver = CBFQPSolver()
    P = np.eye(2)
    q = np.ones(2)

    def test_optimize_optimal1(self) -> None:
        """one inequality constraint"""
        G_list = [np.ones(2)]
        alpha_h_list = [1.0]

        status, optimal_input = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)
        assert status == "optimal"
        assert np.allclose(optimal_input, np.array([-1.00188955, -1.00188955]))

    def test_optimize_optimal2(self) -> None:
        """two inequality constraints"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0, 1.0]

        status, optimal_input = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)
        assert status == "optimal"
        assert np.allclose(optimal_input, np.array([-0.2, -0.2]))

    def test_optimize_except(self) -> None:
        """exception"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0]

        assert len(G_list) != len(alpha_h_list)
        with pytest.raises(Exception) as e:
            _ = self.qp_solver.optimize(self.P, self.q, G_list, alpha_h_list)
        assert str(e.value) == "'G' must be a 'd' matrix of size (1, 2)"


class TestCBFNomQPSolver:
    nom_qp_solver = CBFNomQPSolver()
    P = np.eye(2)
    nominal_input = np.ones(2)

    def test_optimize_optimal1(self) -> None:
        """one inequality constraint"""
        G_list = [np.ones(2)]
        alpha_h_list = [1.0]

        status, optimal_input = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)

        assert status == "optimal"
        assert np.allclose(optimal_input, np.array([1.00188955, 1.00188955]))

    def test_optimize_optimal2(self) -> None:
        """two inequality constraints"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0, 1.0]

        status, optimal_input = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)

        assert status == "optimal"
        assert np.allclose(optimal_input, np.array([0.2, 0.2]))

    def test_optimize_except(self) -> None:
        """exception"""
        G_list = [np.ones(2), -np.ones(2)]
        alpha_h_list = [1.0]

        assert len(G_list) != len(alpha_h_list)
        with pytest.raises(Exception) as e:
            _ = self.nom_qp_solver.optimize(self.nominal_input, self.P, G_list, alpha_h_list)
        assert str(e.value) == "'G' must be a 'd' matrix of size (1, 2)"

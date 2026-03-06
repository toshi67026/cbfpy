#!/usr/bin/env python

from collections.abc import Sequence

from numpy.typing import NDArray

from cbfpy.cbf import CBFBase
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFController:
    """High-level controller that combines multiple CBFs with a QP solver.

    Collects constraints from a list of CBFs and solves the CBF-QP to produce
    a safe input that tracks the given nominal input.

    Args:
        cbf_list: list of CBF instances (constraints must be pre-computed via calc_constraints)
        P: weight matrix for the QP cost. shape=(N, N)

    Example:
        >>> import numpy as np
        >>> from cbfpy import CBFController, CircleCBF
        >>>
        >>> cbf = CircleCBF(center=np.zeros(2), radius=2.0, keep_inside=True)
        >>> controller = CBFController([cbf], P=np.eye(2))
        >>>
        >>> # Update constraints based on current state
        >>> cbf.calc_constraints(agent_position=np.array([1.5, 0.0]))
        >>>
        >>> # Get safe input
        >>> status, safe_input = controller.optimize(nominal_input=np.array([1.0, 0.0]))
    """

    def __init__(self, cbf_list: Sequence[CBFBase], P: NDArray) -> None:
        self._cbf_list = cbf_list
        self._P = P
        self._solver = CBFNomQPSolver()

    @property
    def cbf_list(self) -> Sequence[CBFBase]:
        return self._cbf_list

    @property
    def P(self) -> NDArray:
        return self._P

    @P.setter
    def P(self, value: NDArray) -> None:
        self._P = value

    def optimize(self, nominal_input: NDArray) -> tuple[str, NDArray]:
        """Collect constraints from all CBFs and solve the CBF-QP.

        All CBFs must have their constraints updated (via calc_constraints)
        before calling this method.

        Args:
            nominal_input: desired control input. shape=(N,)

        Returns:
            status: "optimal" on success
            optimal_input: safe control input. shape=(N,)
        """
        G_list: list[NDArray] = []
        alpha_h_list: list[float] = []
        for cbf in self._cbf_list:
            G, alpha_h = cbf.get_constraints()
            G_list.append(G)
            alpha_h_list.append(alpha_h)

        return self._solver.optimize(nominal_input, self._P, G_list, alpha_h_list)

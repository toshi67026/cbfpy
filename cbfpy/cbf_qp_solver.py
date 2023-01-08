#!/usr/bin/env python

from typing import Any, List, Tuple

import cvxopt
import numpy as np
from numpy.typing import NDArray


class QPSolver:
    def __init__(self) -> None:
        self._set_solver_options()

    def _set_solver_options(
        self,
        show_progress: bool = False,
        maxiters: int = 100,
        abstol: float = 1e1,
        reltol: float = 1e1,
        feastol: float = 1e-7,
        refinement: int = 0,
    ) -> None:
        """
        Args:
            show_progress (bool, optional): Turns the output to the screen on or off. Defaults to False.
            maxiters (int, optional): Maximum number of iterations. Defaults to 100.
            abstol (float, optional): Absolute accuracy. Defaults to 1e1.
            reltol (float, optional): Relative accuracy. Defaults to 1e1.
            feastol (float, optional): Tolerance for feasibility conditions. Defaults to 1e-7.
            refinement (int, optional):
                Number of iterative refinement steps when solving KKT equations
                (default: 0 if the problem has no second-order cone or matrix inequality constraints; 1 otherwise)..
                Defaults to 0.
        """
        cvxopt.solvers.options["show_progress"] = show_progress
        cvxopt.solvers.options["maxiters"] = maxiters
        cvxopt.solvers.options["abstol"] = abstol
        cvxopt.solvers.options["reltol"] = reltol
        cvxopt.solvers.options["feastol"] = feastol
        cvxopt.solvers.options["refinement"] = refinement

    def _set_qp_solvers(self, P: NDArray, q: NDArray, G: NDArray, alpha_h: NDArray) -> Any:
        """
        Args:
            P (NDArray): optimization weight matrix. shape=(N, N)
            q (NDArray): optimization weight vector. shape=(N, 1)
            G (NDArray): constraint matrix. shape=(M, N)
            alpha_h (NDArray): constraint vector. shape=(M,)

        Returns:
            Dict[str, List[Any]]: cvxopt.solvers.coneqp
        """
        P_mat = cvxopt.matrix(P.astype("float"))
        q_mat = cvxopt.matrix(q.astype("float"))
        G_mat = cvxopt.matrix(G.astype("float"))
        alpha_h_mat = cvxopt.matrix(alpha_h.astype("float"))

        try:
            return cvxopt.solvers.coneqp(P_mat, q_mat, G_mat, alpha_h_mat)
        except Exception as e:
            raise e


class CBFQPSolver(QPSolver):
    def optimize(
        self,
        P: NDArray,
        q: NDArray,
        G_list: List[NDArray],
        alpha_h_list: List[float],
    ) -> Tuple[str, NDArray]:
        """
        Solve the following optimization problem
            minimize_{u} (1/2) * u^T*P*u + q^T*u
            subject to G*u + alpha(h) >= 0

        Args:
            P (NDArray): optimization weight matrix. shape=(N, N)
            q (NDArray): optimization weight vector. shape=(N, 1)
            G (List[NDArray]): constraint matrix list. [(N,), (N,), ...]
            alpha_h (List[float]): constraint value list

        Notes:
            cvxopt.matrix()にshapeが(N,)のNDArrayを渡すと，shapeが(N, 1)のmatrixを返す．
            ソルバーに与えたいGの構造は(1, N)であるから，matrixにarrayを渡す際に(1, N)のshapeを持つようにしておく必要がある．
            制約が複数あり，ソルバーに渡すGが(M, N)の構造を持つ場合も考慮しないといけないので，以下の通りlistの長さをもとに場合分けして，様々な入力に対して適切なG_matを生成できるよう工夫している．

        Returns:
            (str): status
            (NDArray): optimal input. shape=(N,)
        """
        assert isinstance(G_list, list)
        assert isinstance(alpha_h_list, list)

        if len(G_list) > 1:
            G = np.array(list(map(lambda x: x.flatten(), G_list)))
        else:
            G = G_list[0].reshape(1, -1)
        alpha_h = np.array(alpha_h_list)

        try:
            sol = self._set_qp_solvers(P, q, G, alpha_h)
            return sol["status"], np.array(sol["x"]).flatten()
        except Exception as e:
            raise e


class CBFNomQPSolver(QPSolver):
    def optimize(
        self,
        nominal_input: NDArray,
        P: NDArray,
        G_list: List[NDArray],
        alpha_h_list: List[float],
    ) -> Tuple[str, NDArray]:
        """
        Solve the following optimization problem
            minimize_{u} (1/2) * (u-nominal_input)^T*P*(u-nominal_input)
            subject to G*u + alpha(h) >= 0

        Args:
            nominal_input (NDArray): nominal_input. shape=(N,)
            P (NDArray): optimization weight matrix. shape=(N, N)
            G (List[NDArray]): constraint matrix list. [shape]=[(N,), (N,), ...]
            alpha_h (List[float]): constraint value list

        Returns:
            (str): status
            (NDArray): optimal input. shape=(N,)
        """
        nominal_input = nominal_input.reshape(-1, 1)
        q = -P.T @ nominal_input
        assert isinstance(G_list, list)
        assert isinstance(alpha_h_list, list)

        if len(G_list) > 1:
            G = -np.array(list(map(lambda x: x.flatten(), G_list)))
        else:
            G = -G_list[0].reshape(1, -1)

        alpha_h = np.array(alpha_h_list)

        sol = self._set_qp_solvers(P, q, G, alpha_h)
        return sol["status"], np.array(sol["x"]).flatten()

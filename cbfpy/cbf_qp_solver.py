#!/usr/bin/env python


from collections.abc import Sequence

import numpy as np
import quadprog
from numpy.typing import NDArray


def _assemble_constraints(G_list: Sequence[NDArray]) -> NDArray:
    """Assemble constraint matrix from a list of constraint vectors.

    Args:
        G_list: list of constraint vectors. [(N,), (N,), ...]

    Returns:
        Assembled constraint matrix. shape=(M, N)
    """
    if len(G_list) > 1:
        return np.array([np.atleast_1d(g).flatten() for g in G_list])
    return np.atleast_1d(G_list[0]).reshape(1, -1)


def _solve_qp(P: NDArray, q: NDArray, G_ineq: NDArray, h_ineq: NDArray) -> NDArray:
    """Solve a quadratic program.

        minimize_{x} (1/2) * x^T P x + q^T x
        subject to   G_ineq @ x <= h_ineq

    Uses quadprog (Goldfarb/Idnani dual algorithm) internally.

    Args:
        P: positive definite weight matrix. shape=(N, N)
        q: linear cost vector. shape=(N,) or (N, 1)
        G_ineq: inequality constraint matrix. shape=(M, N)
        h_ineq: inequality constraint bound. shape=(M,)

    Returns:
        Optimal solution x. shape=(N,)
    """
    # quadprog solves: min 1/2 x^T G x - a^T x, s.t. C^T x >= b
    # Mapping: G=P, a=-q, C^T=-G_ineq (so C=-G_ineq^T), b=-h_ineq
    x, *_ = quadprog.solve_qp(
        P.astype(float),
        -q.flatten().astype(float),
        -G_ineq.T.astype(float),
        -h_ineq.flatten().astype(float),
    )
    return np.asarray(x)


class CBFQPSolver:
    """CBF-QP solver for general quadratic cost."""

    def optimize(
        self,
        P: NDArray,
        q: NDArray,
        G_list: Sequence[NDArray],
        alpha_h_list: list[float],
    ) -> tuple[str, NDArray]:
        """Solve the CBF-QP optimization problem.

            minimize_{u} (1/2) * u^T*P*u + q^T*u
            subject to G*u <= alpha(h)

        Args:
            P: optimization weight matrix. shape=(N, N)
            q: optimization weight vector. shape=(N,) or (N, 1)
            G_list: constraint matrix list. [(N,), (N,), ...]
            alpha_h_list: constraint value list

        Returns:
            status: "optimal" on success
            optimal_input: optimal input. shape=(N,)
        """
        G = _assemble_constraints(G_list)
        alpha_h = np.array(alpha_h_list)
        x = _solve_qp(P, q, G, alpha_h)
        return "optimal", x


class CBFNomQPSolver:
    """CBF-QP solver that tracks a nominal input."""

    def optimize(
        self,
        nominal_input: NDArray,
        P: NDArray,
        G_list: Sequence[NDArray],
        alpha_h_list: list[float],
    ) -> tuple[str, NDArray]:
        """Solve the CBF-QP optimization problem with nominal input tracking.

            minimize_{u} (1/2) * (u-nominal_input)^T*P*(u-nominal_input)
            subject to G*u + alpha(h) >= 0

        Args:
            nominal_input: nominal input. shape=(N,)
            P: optimization weight matrix. shape=(N, N)
            G_list: constraint matrix list. [(N,), (N,), ...]
            alpha_h_list: constraint value list

        Returns:
            status: "optimal" on success
            optimal_input: optimal input. shape=(N,)
        """
        nominal_input = nominal_input.reshape(-1, 1)
        q = -P.T @ nominal_input

        G = -_assemble_constraints(G_list)
        alpha_h = np.array(alpha_h_list)
        x = _solve_qp(P, q, G, alpha_h)
        return "optimal", x

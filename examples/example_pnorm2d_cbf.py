#!/usr/bin/env python

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import Pnorm2dCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(2)

        self.pnorm2d_cbf = Pnorm2dCBF()

        # initialize(must be overwritten)
        self.set_parameters(np.zeros(2), np.ones(2))

    def set_parameters(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        self.pnorm2d_cbf.set_parameters(center, width, theta, p, keep_inside)

    def get_parameters(self) -> Tuple[NDArray, NDArray, float, float, bool]:
        return self.pnorm2d_cbf.get_parameters()

    def _calc_constraints(self, agent_position: NDArray) -> None:
        self.pnorm2d_cbf.calc_constraints(agent_position)

    def _get_constraints(self) -> Tuple[List[NDArray], List[float]]:
        G, alpha_h = self.pnorm2d_cbf.get_constraints()
        return [G], [alpha_h]

    def optimize(self, nominal_input: NDArray, agent_position: NDArray) -> Tuple[str, NDArray]:
        self._calc_constraints(agent_position)
        G_list, alpha_h_list = self._get_constraints()

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, G_list, alpha_h_list)
        except Exception as e:
            raise e


def main() -> None:
    optimizer = CBFOptimizer()

    initial_position = np.array([-3, -2.5])
    agent_position_list: List[NDArray] = [initial_position]
    dt = 0.1

    # obstacle
    center = np.array([-0.5, 0.5])
    width = np.array([3, 2])
    theta = -0.3
    p = 2.0
    keep_inside = False

    optimizer.set_parameters(center, width, theta, p, keep_inside)
    nominal_input = np.ones(2)

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_position_list: List[NDArray],
    ) -> None:
        ax.cla()

        curr_position = agent_position_list[-1]
        _, optimal_input = optimizer.optimize(nominal_input, curr_position)
        agent_position_list.append(curr_position + dt * optimal_input)

        # show area
        r = patches.Ellipse(
            xy=center,
            width=width[0] * 2,
            height=width[1] * 2,
            angle=theta * 180 / np.pi,
            color="green",
            alpha=0.5,
            label="keep_inside: " + str(keep_inside),
        )
        ax.add_patch(r)

        ax.plot(curr_position[0], curr_position[1], "o", linewidth=5, label="agent")

        # show vector
        ax.quiver(curr_position[0], curr_position[1], nominal_input[0], nominal_input[1], scale=10, color="black")
        ax.quiver(curr_position[0], curr_position[1], optimal_input[0], optimal_input[1], scale=10, color="red")

        ax.plot([0], [0], linewidth=5, color="black", label="nominal_input")
        ax.plot([0], [0], linewidth=5, color="red", label="optimal_input")

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend(loc="upper right")

        lim = [-5, 5]
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ani = FuncAnimation(
        fig,
        update,
        frames=100,
        fargs=(agent_position_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_pnorm2d_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import CircleCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(2)

        self.circle_cbf = CircleCBF()

        # initialize(must be overwritten)
        self.set_parameters(np.zeros(2))

    def set_parameters(self, center: NDArray, radius: float = 1.0, keep_inside: bool = True) -> None:
        self.circle_cbf.set_parameters(center, radius, keep_inside)

    def get_parameters(self) -> Tuple[NDArray, float, bool]:
        return self.circle_cbf.get_parameters()

    def _calc_constraints(self, agent_position: NDArray) -> None:
        self.circle_cbf.calc_constraints(agent_position)

    def _get_constraints(self) -> Tuple[List[NDArray], List[float]]:
        G, alpha_h = self.circle_cbf.get_constraints()
        return [G], [alpha_h]

    def optimize(self, nominal_input: NDArray, agent_position: NDArray) -> Tuple[str, NDArray]:
        self._calc_constraints(agent_position)
        G_list, alpha_h_list = self._get_constraints()

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, G_list, alpha_h_list)
        except Exception as e:
            raise e


def main() -> None:
    optimizer_list = [CBFOptimizer(), CBFOptimizer()]

    initial_position_array = np.array([[-2, -2.5], [2, 2]])
    agent_position_list: List[NDArray] = [initial_position_array]
    dt = 0.1

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_position_list: List[NDArray],
    ) -> None:
        ax.cla()

        curr_position_array = agent_position_list[-1]

        for agent_id in range(2):
            # another agent position
            another_agent_position = curr_position_array[~agent_id]
            radius = 1.0

            keep_inside = False

            optimizer_list[agent_id].set_parameters(another_agent_position, 2 * radius, keep_inside)
            nominal_input = np.array([-2 * agent_id + 1] * 2)
            curr_position = curr_position_array[agent_id]
            _, optimal_input = optimizer_list[agent_id].optimize(nominal_input, curr_position)

            curr_position_array[agent_id] = curr_position_array[agent_id] + dt * optimal_input

            # show area
            r = patches.Circle(
                xy=another_agent_position,
                radius=radius,
                color="green",
                alpha=0.5,
            )
            ax.add_patch(r)

            ax.plot(curr_position[0], curr_position[1], "o", linewidth=5, label="agent" + str(agent_id))

            # show vector
            ax.quiver(curr_position[0], curr_position[1], nominal_input[0], nominal_input[1], scale=10, color="black")
            ax.quiver(curr_position[0], curr_position[1], optimal_input[0], optimal_input[1], scale=10, color="red")

        agent_position_list.append(curr_position_array)

        ax.plot([0], [0], linewidth=5, color="black", label="nominal_input")
        ax.plot([0], [0], linewidth=5, color="red", label="optimal_input")
        ax.plot([0], [0], linewidth=5, color="green", alpha=0.5, label="keep_inside: " + str(keep_inside))

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend(loc="upper left")

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

    # ani.save("example_circle_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

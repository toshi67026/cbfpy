#!/usr/bin/env python

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import UnicycleCircleCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(2)

        self.circle_cbf = UnicycleCircleCBF()

        # initialize(must be overwritten)
        self.set_parameters(np.zeros(2))

    def set_parameters(self, center: NDArray, radius: float = 1.0, keep_inside: bool = True) -> None:
        self.circle_cbf.set_parameters(center, radius, keep_inside)

    def get_parameters(self) -> Tuple[NDArray, float, bool]:
        return self.circle_cbf.get_parameters()

    def _calc_constraints(self, agent_pose: NDArray) -> None:
        self.circle_cbf.calc_constraints(agent_pose)

    def _get_constraints(self) -> Tuple[List[NDArray], List[float]]:
        G, alpha_h = self.circle_cbf.get_constraints()
        return [G], [alpha_h]

    def optimize(self, nominal_input: NDArray, agent_pose: NDArray) -> Tuple[str, NDArray]:
        self._calc_constraints(agent_pose)
        G_list, alpha_h_list = self._get_constraints()

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, G_list, alpha_h_list)
        except Exception as e:
            raise e


def main() -> None:
    optimizer = CBFOptimizer()

    initial_pose_array = np.array([0, 0.5, 0.0])
    agent_pose_list: List[NDArray] = [initial_pose_array]
    dt = 0.1
    center = np.array([-0.5, 0.5])
    radius = 1.5
    keep_inside = True

    optimizer.set_parameters(center, radius, keep_inside)

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_pose_list: List[NDArray],
    ) -> None:
        ax.cla()

        nominal_input = np.array([1.0, 0.3])

        curr_pose = agent_pose_list[-1]

        optimizer.set_parameters(center, radius, keep_inside)
        _, optimal_input = optimizer.optimize(nominal_input, curr_pose)
        curr_position = curr_pose[0:2]
        curr_theta = curr_pose[2]

        transform_matrix: NDArray = np.array(
            [
                [np.cos(curr_theta), 0],
                [0, np.sin(curr_theta)],
                [0, 1],
            ]
        )
        agent_pose_list.append(curr_pose + dt * transform_matrix @ optimal_input)

        # show area
        r = patches.Circle(
            xy=center,
            radius=radius,
            color="green",
            alpha=0.5,
            label="keep_inside: " + str(keep_inside),
        )
        ax.add_patch(r)

        ax.plot(curr_position[0], curr_position[1], "o", linewidth=5, label="agent")

        # show vector
        # v
        ax.quiver(
            curr_position[0],
            curr_position[1],
            nominal_input[0] * np.cos(curr_theta),
            nominal_input[0] * np.sin(curr_theta),
            scale=10,
            color="black",
        )
        ax.quiver(
            curr_position[0],
            curr_position[1],
            optimal_input[0] * np.cos(curr_theta),
            optimal_input[0] * np.sin(curr_theta),
            scale=10,
            color="red",
        )

        # omega
        ax.quiver(
            curr_position[0],
            curr_position[1],
            -nominal_input[1] * np.sin(curr_theta),
            nominal_input[1] * np.cos(curr_theta),
            scale=10,
            color="black",
        )
        ax.quiver(
            curr_position[0],
            curr_position[1],
            -optimal_input[1] * np.sin(curr_theta),
            optimal_input[1] * np.cos(curr_theta),
            scale=10,
            color="red",
        )

        ax.plot([0], [0], linewidth=5, color="black", label="nominal_input")
        ax.plot([0], [0], linewidth=5, color="red", label="optimal_input")

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend(loc="upper left")

        lim = [-3, 3]
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ani = FuncAnimation(
        fig,
        update,
        frames=100,
        fargs=(agent_pose_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_unicycle_circle_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

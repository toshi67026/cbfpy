#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import UnicyclePnorm2dCBF
from cbfpy.cbf_controller import CBFController


def main() -> None:
    center = np.array([-0.5, 0.5])
    width = np.array([3, 2])
    theta = -0.3
    p = 2.0
    keep_inside = True

    cbf = UnicyclePnorm2dCBF(center, width, theta, p, keep_inside)
    controller = CBFController([cbf], P=np.diag([1, 10]))

    initial_pose_array = np.array([-1, -1, 0])
    agent_pose_list: list[NDArray] = [initial_pose_array]
    dt = 0.1

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_pose_list: list[NDArray],
    ) -> None:
        ax.cla()

        nominal_input = np.array([1.0, 0.2])

        curr_pose = agent_pose_list[-1]

        cbf.set_parameters(center, width, theta, p, keep_inside)
        cbf.calc_constraints(curr_pose)
        _, optimal_input = controller.optimize(nominal_input)
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
        ax.legend(loc="upper right")

        lim = [-5, 5]
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ani = FuncAnimation(  # noqa: F841
        fig,
        update,
        frames=200,
        fargs=(agent_pose_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_unicycle_pnorm2d_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

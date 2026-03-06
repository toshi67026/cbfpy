#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import Pnorm2dCBF
from cbfpy.cbf_controller import CBFController


def main() -> None:
    # obstacle
    center = np.array([-0.5, 0.5])
    width = np.array([3, 2])
    theta = -0.3
    p = 2.0
    keep_inside = False

    cbf = Pnorm2dCBF(center, width, theta, p, keep_inside)
    controller = CBFController([cbf], P=np.eye(2))

    initial_position = np.array([-3, -2.5])
    agent_position_list: list[NDArray] = [initial_position]
    dt = 0.1
    nominal_input = np.ones(2)

    fig, ax = plt.subplots()

    def update(
        frame: int,
        agent_position_list: list[NDArray],
    ) -> None:
        ax.cla()

        curr_position = agent_position_list[-1]
        cbf.calc_constraints(curr_position)
        _, optimal_input = controller.optimize(nominal_input)
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

    ani = FuncAnimation(  # noqa: F841
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

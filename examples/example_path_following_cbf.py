#!/usr/bin/env python

"""Path following with obstacle avoidance using CBFController.

An agent follows a sequence of waypoints while avoiding circular obstacles.
Multiple CircleCBFs (keep_inside=False) enforce collision avoidance.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import CircleCBF
from cbfpy.cbf_controller import CBFController


def main() -> None:
    # Obstacles: (center, radius)
    obstacles = [
        (np.array([1.0, 1.0]), 1.0),
        (np.array([3.5, -0.5]), 0.8),
        (np.array([2.0, 3.0]), 0.7),
        (np.array([5.0, 2.0]), 1.2),
    ]

    # Waypoints for the agent to follow
    waypoints = np.array(
        [
            [0.0, 0.0],
            [2.0, -1.0],
            [4.0, 1.0],
            [6.0, 0.0],
            [6.0, 3.0],
            [3.0, 4.0],
            [0.0, 3.0],
        ]
    )

    # Create CBFs for each obstacle (keep_inside=False = stay outside)
    cbf_list = [CircleCBF(center, radius, keep_inside=False) for center, radius in obstacles]
    controller = CBFController(cbf_list, P=np.eye(2))

    dt = 0.1
    speed = 1.5
    waypoint_threshold = 0.3
    current_wp_idx = 0

    initial_position = waypoints[0].copy()
    position_list: list[NDArray] = [initial_position]

    fig, ax = plt.subplots()

    def update(
        frame: int,
        position_list: list[NDArray],
    ) -> None:
        nonlocal current_wp_idx
        ax.cla()

        curr_position = position_list[-1]

        # Advance waypoint if close enough
        if current_wp_idx < len(waypoints) - 1:
            dist_to_wp = float(np.linalg.norm(waypoints[current_wp_idx + 1] - curr_position))
            if dist_to_wp < waypoint_threshold:
                current_wp_idx = min(current_wp_idx + 1, len(waypoints) - 2)

        # Nominal input: move toward next waypoint
        target = waypoints[current_wp_idx + 1]
        direction = target - curr_position
        dist = float(np.linalg.norm(direction))
        nominal_input = speed * direction / max(dist, 0.1)

        # Update constraints and optimize
        for cbf in cbf_list:
            cbf.calc_constraints(curr_position)
        _, optimal_input = controller.optimize(nominal_input)

        position_list.append(curr_position + dt * optimal_input)

        # Draw obstacles
        for center, radius in obstacles:
            circle = patches.Circle(xy=center, radius=radius, color="green", alpha=0.4)
            ax.add_patch(circle)

        # Draw waypoints and path
        ax.plot(waypoints[:, 0], waypoints[:, 1], "s--", color="gray", alpha=0.5, markersize=6, label="waypoints")

        # Draw trajectory
        traj = np.array(position_list)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:blue", alpha=0.6, linewidth=1.5)
        ax.plot(curr_position[0], curr_position[1], "o", color="tab:blue", markersize=8, label="agent")

        # Show vectors
        ax.quiver(curr_position[0], curr_position[1], nominal_input[0], nominal_input[1], scale=10, color="black")
        ax.quiver(curr_position[0], curr_position[1], optimal_input[0], optimal_input[1], scale=10, color="red")
        ax.plot([], [], linewidth=3, color="black", label="nominal_input")
        ax.plot([], [], linewidth=3, color="red", label="optimal_input")

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        ax.legend(loc="upper left")
        lim_x = [-1, 8]
        lim_y = [-2, 5]
        ax.set_xlim(lim_x)
        ax.set_ylim(lim_y)

    ani = FuncAnimation(  # noqa: F841
        fig,
        update,
        frames=200,
        fargs=(position_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_path_following_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

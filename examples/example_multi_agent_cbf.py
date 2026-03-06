#!/usr/bin/env python

"""Multi-agent collision avoidance using CBFController.

N agents move toward their goals while avoiding each other.
Each agent has N-1 CircleCBFs (one per other agent, keep_inside=False).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import CircleCBF
from cbfpy.cbf_controller import CBFController


def main() -> None:
    num_agents = 4
    safe_radius = 1.0

    # Initial positions (square formation)
    positions = np.array(
        [
            [-3.0, -3.0],
            [3.0, -3.0],
            [3.0, 3.0],
            [-3.0, 3.0],
        ]
    )
    # Goals (swap diagonally)
    goals = np.array(
        [
            [3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, -3.0],
            [3.0, -3.0],
        ]
    )

    # Each agent has CBFs for all other agents
    cbf_lists: list[list[CircleCBF]] = []
    controllers: list[CBFController] = []
    for i in range(num_agents):
        cbfs = [CircleCBF(positions[j], 2 * safe_radius, keep_inside=False) for j in range(num_agents) if j != i]
        cbf_lists.append(cbfs)
        controllers.append(CBFController(cbfs, P=np.eye(2)))

    dt = 0.1
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots()
    position_history: list[NDArray] = [positions.copy()]

    def update(frame: int, position_history: list[NDArray]) -> None:
        ax.cla()
        curr_positions = position_history[-1].copy()

        for i in range(num_agents):
            # Update CBF parameters (other agents' positions)
            other_idx = 0
            for j in range(num_agents):
                if j == i:
                    continue
                cbf_lists[i][other_idx].set_parameters(curr_positions[j], 2 * safe_radius, keep_inside=False)
                cbf_lists[i][other_idx].calc_constraints(curr_positions[i])
                other_idx += 1

            # Nominal input: move toward goal
            direction = goals[i] - curr_positions[i]
            dist = np.linalg.norm(direction)
            speed = 2.0
            nominal_input = speed * direction / max(dist, 0.1)

            _, optimal_input = controllers[i].optimize(nominal_input)
            curr_positions[i] = curr_positions[i] + dt * optimal_input

            # Draw
            circle = patches.Circle(xy=curr_positions[i], radius=safe_radius, color=colors[i], alpha=0.2)
            ax.add_patch(circle)
            ax.plot(curr_positions[i][0], curr_positions[i][1], "o", color=colors[i], markersize=8)
            ax.plot(goals[i][0], goals[i][1], "x", color=colors[i], markersize=10, markeredgewidth=2)

            # Show trajectory
            traj = np.array([p[i] for p in position_history])
            ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], alpha=0.4, linewidth=1)

        position_history.append(curr_positions)

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid()
        lim = [-5, 5]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"Multi-Agent CBF (t={frame * dt:.1f}s)")

    ani = FuncAnimation(  # noqa: F841
        fig,
        update,
        frames=150,
        fargs=(position_history,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_multi_agent_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

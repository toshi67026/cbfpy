#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import ScalarCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self, limit: float, keep_upper: bool = True) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(1)
        self.scalar_cbf = ScalarCBF(limit, keep_upper)

    def set_parameters(self, limit: float, keep_upper: bool = True) -> None:
        self.scalar_cbf.set_parameters(limit, keep_upper)

    def get_parameters(self) -> tuple[float, bool]:
        return self.scalar_cbf.get_parameters()

    def optimize(self, nominal_input: float, curr_value: float) -> tuple[str, NDArray]:
        self.scalar_cbf.calc_constraints(curr_value)
        G, alpha_h = self.scalar_cbf.get_constraints()
        return self.qp_nom_solver.optimize(np.array(nominal_input), self.P, [G], [alpha_h])


def main() -> None:
    optimizer = CBFOptimizer(limit=0.0)

    initial_value = 0.0
    value_list: list[float] = [initial_value]
    time_list: list[float] = [0.0]
    dt = 0.1

    fig, ax = plt.subplots()

    def update(
        frame: int,
        value_list: list[float],
        time_list: list[float],
    ) -> None:
        ax.cla()

        curr_time = time_list[-1]
        if curr_time < 3:
            limit = -1.0
            nominal_input = -1.0
            keep_upper = True
        elif 3 <= curr_time < 6:
            limit = -0.5
            nominal_input = 1.0
            keep_upper = False
        else:
            limit = -1.5
            nominal_input = -1.0
            keep_upper = True

        optimizer.set_parameters(limit, keep_upper)
        curr_value = value_list[-1]
        _, optimal_input = optimizer.optimize(nominal_input, curr_value)
        value_list.append(curr_value + float(optimal_input) * dt)
        time_list.append(curr_time + dt)

        # show limit line
        xlim = [0, 10]
        ax.hlines(
            limit,
            *xlim,
            colors="green",
            linestyles="dashed",
            linewidth=3,
            label="keep_upper" if keep_upper else "keep_lower",
        )

        # show value
        ax.plot(time_list, value_list, linewidth=3, label="value")

        # show vector
        ax.quiver(time_list[-1], value_list[-1], 0, nominal_input, scale=10, color="black")
        ax.quiver(time_list[-1], value_list[-1], 0, optimal_input, scale=10, color="red")
        ax.plot([0], [0], linewidth=5, color="black", label="nominal_input")
        ax.plot([0], [0], linewidth=5, color="red", label="optimal_input")

        ax.set_aspect("equal")
        ax.set_xlabel("Time [s]")
        ax.grid()

        ax.set_xlim(xlim)
        ax.set_ylim([-3, 3])
        ax.legend(loc="upper right")

    ani = FuncAnimation(  # noqa: F841
        fig,
        update,
        frames=100,
        fargs=(
            value_list,
            time_list,
        ),
        interval=10,
        repeat=False,
    )

    # ani.save("example_scalar_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

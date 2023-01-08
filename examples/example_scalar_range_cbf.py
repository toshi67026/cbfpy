#!/usr/bin/env python

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import ScalarRangeCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        self.P = np.eye(1)

        self.scalar_range_cbf = ScalarRangeCBF()

        # initialize (must be overwritten)
        self.set_parameters(0.0, 1.0)

    def set_parameters(self, a: float, b: float, keep_inside: bool = True) -> None:
        self.scalar_range_cbf.set_parameters(a, b, keep_inside)

    def get_parameters(self) -> Tuple[float, float, bool]:
        return self.scalar_range_cbf.get_parameters()

    def _calc_constraints(self, curr_value: float) -> None:
        self.scalar_range_cbf.calc_constraints(curr_value)

    def _get_constraints(self) -> Tuple[List[NDArray], List[float]]:
        G, alpha_h = self.scalar_range_cbf.get_constraints()
        return [G], [alpha_h]

    def optimize(self, nominal_input: float, curr_value: float) -> Tuple[str, NDArray]:
        self._calc_constraints(curr_value)
        G_list, alpha_h_list = self._get_constraints()

        try:
            return self.qp_nom_solver.optimize(np.array(nominal_input), self.P, G_list, alpha_h_list)
        except Exception as e:
            raise e


def main() -> None:
    optimizer = CBFOptimizer()

    initial_value = 0.0
    value_list: List[float] = [initial_value]
    time_list: List[float] = [0.0]
    dt = 0.1

    fig, ax = plt.subplots()

    def update(
        frame: int,
        value_list: List[float],
        time_list: List[float],
    ) -> None:
        ax.cla()

        curr_time = time_list[-1]
        if curr_time < 3:
            a = -2.0
            b = -1.0
            nominal_input = -1.0
            keep_inside = True
        elif 3 <= curr_time < 6:
            a = -0.5
            b = 1.0
            nominal_input = 1.0
            keep_inside = False
        else:
            a = -2.0
            b = -1.0
            nominal_input = -1.0
            keep_inside = True

        optimizer.set_parameters(a, b, keep_inside)
        curr_value = value_list[-1]
        _, optimal_input = optimizer.optimize(nominal_input, curr_value)
        value_list.append(curr_value + float(optimal_input) * dt)
        time_list.append(curr_time + dt)

        # show area
        ax.axhspan(
            a,
            b,
            0,
            1,
            color="green",
            linewidth=3,
            alpha=0.5,
            label="keep_inside" if keep_inside else "keep_outside",
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

        xlim = [0, 10]
        ax.set_xlim(xlim)
        ax.set_ylim([-3, 3])
        ax.legend(loc="upper right")

    ani = FuncAnimation(
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

    # ani.save("example_scalar_range_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

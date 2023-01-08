#!/usr/bin/env python

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from cbfpy.cbf import LiDARCBF
from cbfpy.cbf_qp_solver import CBFNomQPSolver


class CBFOptimizer:
    def __init__(self, num_points: int) -> None:
        self.qp_nom_solver = CBFNomQPSolver()
        # Set weights so that angular velocity is more likely to occur.
        self.P = np.diag([10, 1])

        self.lidar_cbf_list = [LiDARCBF()] * num_points

    def set_parameters(self, width: NDArray, keep_upper: bool = True) -> None:
        for lidar_cbf in self.lidar_cbf_list:
            lidar_cbf.set_parameters(width, keep_upper)

    def get_parameters(self) -> List[Tuple[NDArray, bool]]:
        return [lidar_cbf.get_parameters() for lidar_cbf in self.lidar_cbf_list]

    def calc_constraints(self, r: NDArray, theta: NDArray) -> None:
        for i, lidar_cbf in enumerate(self.lidar_cbf_list):
            lidar_cbf.calc_constraints(r[i], theta[i])

    def get_constraints(self) -> Tuple[List[NDArray], List[float]]:
        G_list: List[NDArray] = []
        alpha_h_list: List[float] = []
        for lidar_cbf in self.lidar_cbf_list:
            G, alpha_h = lidar_cbf.get_constraints()
            G_list.append(G)
            alpha_h_list.append(alpha_h)

        return G_list, alpha_h_list

    def optimize(self, nominal_input: NDArray, r: NDArray, theta: NDArray) -> Tuple[str, NDArray]:
        self.calc_constraints(r, theta)
        G_list, alpha_h_list = self.get_constraints()

        try:
            return self.qp_nom_solver.optimize(nominal_input, self.P, G_list, alpha_h_list)
        except Exception as e:
            raise e


class Obstacle:
    @abstractmethod
    def is_inner(self, r: float, theta: float, curr_pose: NDArray) -> bool:
        ...

    @abstractmethod
    def plot_func(self, color: str, alpha: float) -> patches:
        ...


@dataclass
class CircleObstacle(Obstacle):
    center: NDArray
    radius: float

    def is_inner(self, r: float, theta: float, curr_pose: NDArray) -> bool:
        return cast(
            bool,
            np.sqrt(
                sum(
                    (
                        np.array([r * np.cos(theta + curr_pose[2]), r * np.sin(theta + curr_pose[2])])
                        + np.array(curr_pose[0:2])
                        - self.center
                    )
                    ** 2
                )
            )
            <= self.radius,
        )

    def plot_func(self, color: str, alpha: float) -> patches:
        return patches.Circle(xy=self.center, radius=self.radius, color=color, alpha=alpha)


@dataclass
class RectangleObstacle(Obstacle):
    center: NDArray
    width: NDArray
    theta: float

    def is_inner(self, r: float, theta: float, curr_pose: NDArray) -> bool:
        rotation_matrix = self._get_rotation_matrix(-self.theta)

        # standardization to unit square
        transformed_position: NDArray = (
            rotation_matrix
            @ (
                np.array(
                    [r * np.cos(theta + curr_pose[2]), r * np.sin(theta + curr_pose[2])] + np.array(curr_pose[0:2])
                )
                - self.center
            )
            / self.width
        )

        # max norm
        return cast(bool, np.linalg.norm(transformed_position, ord=np.inf) <= 1)

    def plot_func(self, color: str, alpha: float) -> patches:
        # matplotlib>=3.6 is required
        return patches.Rectangle(
            xy=self.center - self.width,
            width=self.width[0] * 2,
            height=self.width[1] * 2,
            angle=self.theta * 180 / np.pi,
            rotation_point="center",
            color=color,
            alpha=alpha,
        )

    @staticmethod
    def _get_rotation_matrix(rad: float) -> NDArray:
        return np.array(
            [
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)],
            ]
        )


class LiDARSimulator:
    def __init__(
        self,
        num_points: int,
        range_min: float,
        range_max: float,
        range_step_num: int,
        obstacle_list: Sequence[Obstacle],
    ) -> None:
        self.num_points = num_points
        assert 0 < range_min < range_max
        self.range_min = range_min
        self.range_max = range_max
        self.range_step_num = range_step_num
        self.obstacle_list = obstacle_list

    def sim(self, curr_pose: NDArray) -> Tuple[NDArray, NDArray]:
        # in agent coordinate
        theta_array = np.array([2 * np.pi / self.num_points * i for i in range(self.num_points)])
        range_step_array = np.array(
            [
                (self.range_max - self.range_min) / self.range_step_num * i + self.range_min
                for i in range(self.range_step_num)
            ]
        )

        def linear_search(
            range_step_array: NDArray,
            theta: float,
            curr_pose: NDArray,
            obstacle_list: Sequence[Obstacle],
        ) -> float:
            """Linear search for measured distance

            Args:
                range_step_array (NDArray): candidate sequence of measured distances
                theta (float): laser angle in agent coordinate
                curr_pose (NDArray): current pose(x,y,theta)
                obstacle_list (Sequence[Obstacle]): obstacles

            Returns:
                float: measured distance
            """
            for range_ in range_step_array:
                if any([obstacle.is_inner(range_, theta, curr_pose) for obstacle in obstacle_list]):
                    return float(range_)
            else:
                return self.range_max

        range_array = np.array(
            [
                linear_search(range_step_array, theta_array[i], curr_pose, self.obstacle_list)
                for i in range(self.num_points)
            ]
        )

        return range_array, theta_array


def main() -> None:
    num_points = 20
    optimizer = CBFOptimizer(num_points)
    optimizer.set_parameters(np.array([0.7, 0.5]))

    initial_pose = np.array([0, -3, -0.1])
    agent_pose_list: List[NDArray] = [initial_pose]
    dt = 0.1

    # set obstacles
    obstacle_list = [
        RectangleObstacle(np.array([2, -4]), np.array([2, 0.5]), 0.5),
        RectangleObstacle(np.array([4, 0]), np.array([0.5, 3]), 0),
        CircleObstacle(np.array([3.5, 3]), 1),
        CircleObstacle(np.array([-1, 5]), 2),
        CircleObstacle(np.array([0, 0]), 1.5),
        RectangleObstacle(np.array([-3, -1]), np.array([0.5, 3]), 0.3),
        CircleObstacle(np.array([-2, -4]), 1),
    ]
    lidar_sim = LiDARSimulator(num_points, 0.3, 2.0, 50, obstacle_list)

    fig, ax = plt.subplots()

    def update(frame: int, agent_pose_list: List[NDArray]) -> None:
        ax.cla()

        curr_pose = agent_pose_list[-1]
        curr_position = curr_pose[0:2]
        curr_theta = curr_pose[2]

        nominal_input = np.array([0.5, 0])

        # agent coordinate
        r, theta = lidar_sim.sim(curr_pose)

        _, optimal_input = optimizer.optimize(nominal_input, r, theta)
        transform_matrix: NDArray = np.array(
            [
                [np.cos(curr_theta), 0],
                [np.sin(curr_theta), 0],
                [0, 1],
            ]
        )
        agent_pose_list.append(curr_pose + dt * transform_matrix @ optimal_input)

        # show laser rays
        for i in range(lidar_sim.num_points):
            if lidar_sim.range_min < r[i] < lidar_sim.range_max:
                style = "b-"
            elif r[i] == lidar_sim.range_max:
                style = "y-"
            else:
                style = "g-"

            ax.plot(
                [
                    curr_position[0] + lidar_sim.range_min * np.cos(theta[i] + curr_theta),
                    curr_position[0] + r[i] * np.cos(theta[i] + curr_theta),
                ],
                [
                    curr_position[1] + lidar_sim.range_min * np.sin(theta[i] + curr_theta),
                    curr_position[1] + r[i] * np.sin(theta[i] + curr_theta),
                ],
                style,
                marker=".",
                markeredgewidth=0,
            )

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

        param_list = optimizer.get_parameters()
        width = param_list[0][0]
        r = patches.Ellipse(
            xy=curr_position,
            width=width[0] * 2,
            height=width[1] * 2,
            angle=curr_theta * 180 / np.pi,
            color="blue",
            alpha=0.5,
        )
        ax.add_patch(r)
        for obstacle in obstacle_list:
            ax.add_patch(obstacle.plot_func(color="green", alpha=0.5))

        ax.set_aspect("equal")
        ax.grid()

        lim = [-5, 5]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend(loc="upper left")

    ani = FuncAnimation(
        fig,
        update,
        frames=500,
        fargs=(agent_pose_list,),
        interval=10,
        repeat=False,
    )

    # ani.save("example_lidar_cbf.gif")
    plt.show()


if __name__ == "__main__":
    main()

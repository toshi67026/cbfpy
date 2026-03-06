#!/usr/bin/env python

from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, Symbol, cos, lambdify, sin, sqrt, symbols


def rotation_matrix_2d(rad: float) -> NDArray:
    """Create a 2D rotation matrix.

    Args:
        rad: rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    return np.array(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ]
    )


class CBFBase:
    """CBF base class

    The CBF optimization problem is formulated as:
        minimize_{u} {objective function}
        subject to G*u + alpha(h) >= 0

    Attributes:
        G: constraint matrix (=dh/dx)
        h: constraint value (=h(x))
    """

    G: NDArray
    h: float

    def get_constraints(self) -> Tuple[NDArray, float]:
        """
        Returns:
            G: constraint matrix
            alpha(h): value of alpha function applied to h
        """
        return self.G, self._alpha(self.h)

    def _alpha(self, h: float) -> float:
        """Extended class-K function. Override to customize.

        Args:
            h: constraint value (=h(x))

        Returns:
            alpha(h). Default implementation returns h itself.
        """
        return h


class GeneralCBF(CBFBase):
    """General-purpose CBF with manually specified constraints.

    Attributes:
        G: constraint matrix (=dh/dx). shape=(N,)
        h: constraint value (=h(x))
    """

    def __init__(self, G: NDArray, h: float) -> None:
        self.G = G
        self.h = h

    def calc_constraints(self, G: NDArray, h: float) -> None:
        self.G = G
        self.h = h


class ScalarCBF(CBFBase):
    """CBF for scalar state variable.

    If keep_upper is True, the safety set is x - limit >= 0.

    Attributes:
        limit: limit for scalar state variable
        keep_upper: flag to prohibit going lower of the limit
        G: constraint matrix (=dh/dx). shape=(1,)
        h: constraint value (=h(x))
    """

    def __init__(self, limit: float, keep_upper: bool = True) -> None:
        self.x = Symbol("x", real=True)  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(limit, keep_upper)

    def set_parameters(self, limit: float, keep_upper: bool = True) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.limit = limit
        self.keep_upper = keep_upper

        cbf = self.sign * (self.x - self.limit)
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def get_parameters(self) -> Tuple[float, bool]:
        return self.limit, self.keep_upper

    def calc_constraints(self, curr_value: float) -> None:
        sign = 1 if self.keep_upper else -1

        self.G = self._calc_dhdx(curr_value, sign)
        self.h = self._calc_h(curr_value, sign)


class ScalarRangeCBF(CBFBase):
    """CBF for scalar state variable with range constraint.

    If keep_inside is True, the safety set is a <= x <= b.

    Attributes:
        a: lower limit for scalar state variable
        b: upper limit for scalar state variable
        keep_inside: flag to prohibit going outside of the range
        G: constraint matrix (=dh/dx). shape=(1,)
        h: constraint value (=h(x))
    """

    def __init__(self, a: float, b: float, keep_inside: bool = True) -> None:
        self.x = Symbol("x", real=True)  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(a, b, keep_inside)

    def set_parameters(self, a: float, b: float, keep_inside: bool = True) -> None:
        """Set parameters and rebuild symbolic functions."""
        assert a < b
        self.a = a
        self.b = b
        self.keep_inside = keep_inside

        cbf = self.sign * (((self.b - self.a) / 2) ** 2 - (self.x - (self.a + self.b) / 2) ** 2)
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def get_parameters(self) -> Tuple[float, float, bool]:
        return self.a, self.b, self.keep_inside

    def calc_constraints(self, curr_value: float) -> None:
        sign = 1 if self.keep_inside else -1

        self.G = self._calc_dhdx(curr_value, sign)
        self.h = self._calc_h(curr_value, sign)


class CircleCBF(CBFBase):
    """CBF for circular area constraint.

    Attributes:
        center: center of circular area in world coordinate. shape=(2,)
        radius: radius of the circular area
        keep_inside: flag to prohibit going outside of the area
        G: constraint matrix (=dh/dx). shape=(2,)
        h: constraint value (=h(x))
    """

    def __init__(self, center: NDArray, radius: float, keep_inside: bool = True) -> None:
        self.x = Matrix(symbols("x, y", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(center, radius, keep_inside)

    def set_parameters(self, center: NDArray, radius: float, keep_inside: bool = True) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.center = center.flatten()
        assert radius > 0
        self.radius = radius
        self.keep_inside = keep_inside

        cbf = self.sign * (1.0 - self.x.norm(ord=2))  # type: ignore
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def get_parameters(self) -> Tuple[NDArray, float, bool]:
        return self.center, self.radius, self.keep_inside

    def calc_constraints(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position: agent position in world coordinate. shape=(2,)

        Note:
            Division by radius is a remnant of the transformation.
        """
        agent_position_transformed = self._transform_agent_position(agent_position.flatten())
        sign = 1 if self.keep_inside else -1

        self.G = (self._calc_dhdx(agent_position_transformed, sign) / self.radius).flatten()
        assert self.G.shape == (2,)
        self.h = self._calc_h(agent_position_transformed, sign)

    def _transform_agent_position(self, agent_position: NDArray) -> NDArray:
        """Transform agent position from world coordinate to unit circle.

        Args:
            agent_position: agent position in world coordinate. shape=(2,)

        Returns:
            Transformed agent position within unit circle. shape=(2,)
        """
        return cast(NDArray, ((agent_position - self.center) / self.radius))


class UnicycleCircleCBF(CircleCBF):
    """CBF for circular area constraint with unicycle model.

    Attributes:
        center: center of circular area in world coordinate. shape=(2,)
        radius: radius of the circular area
        keep_inside: flag to prohibit going outside of the area
        G: constraint matrix (=dh/dx). shape=(2,)
        h: constraint value (=h(x))
    """

    def __init__(self, center: NDArray, radius: float, keep_inside: bool = True) -> None:
        self.x = Matrix(symbols("x, y, theta", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(center, radius, keep_inside)

    def set_parameters(self, center: NDArray, radius: float, keep_inside: bool = True) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.center = center.flatten()
        assert radius > 0
        self.radius = radius
        self.keep_inside = keep_inside

        cbf = self.sign * (1.0 - (np.hstack([np.eye(2), np.zeros([2, 1])]) @ self.x).norm(ord=2))
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def calc_constraints(self, agent_pose: NDArray) -> None:
        """
        Args:
            agent_pose: agent pose in world coordinate. shape=(3,) [x, y, theta]

        Note:
            Division by radius is a remnant of the transformation.
        """
        sign = 1 if self.keep_inside else -1
        agent_pose_transformed = self._transform_agent_pose(agent_pose.flatten())

        theta = agent_pose[2]
        A = np.array(
            [
                [np.cos(theta), 0],
                [np.sin(theta), 0],
                [0, 1],
            ]
        )
        self.G = ((self._calc_dhdx(agent_pose_transformed, sign) / self.radius).reshape(1, -1) @ A).flatten()
        assert self.G.shape == (2,)
        self.h = self._calc_h(agent_pose_transformed, sign)

    def _transform_agent_pose(self, agent_pose: NDArray) -> NDArray:
        """Transform agent pose from world coordinate to unit circle.

        Args:
            agent_pose: agent pose in world coordinate. shape=(3,)

        Returns:
            Transformed agent pose within unit circle. shape=(3,)
        """
        agent_position = agent_pose[0:2]
        return cast(NDArray, np.append((agent_position - self.center) / self.radius, agent_pose[2]))


class Pnorm2dCBF(CBFBase):
    """CBF for p-norm shaped area constraint.

    Attributes:
        center: center of area in world coordinate. shape=(2,)
        width: half length of the major and minor axis of ellipse
            that match the x and y axis in the area coordinate. shape=(2,)
        theta: rotation angle (rad) from world to area coordinate
        p: multiplier for p-norm
        keep_inside: flag to prohibit going outside of the area
        G: constraint matrix (=dh/dx). shape=(2,)
        h: constraint value (=h(x))
    """

    def __init__(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        self.x = Matrix(symbols("x, y", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(center, width, theta, p, keep_inside)

    def set_parameters(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.center = center.flatten()
        self.width = width.flatten()
        self.theta = theta
        assert p >= 1
        self.p = p
        self.keep_inside = keep_inside

        cbf = self.sign * (1.0 - sum(abs(self.x.applyfunc(lambda x: x**self.p))) ** (1 / self.p))  # type: ignore
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def get_parameters(self) -> Tuple[NDArray, NDArray, float, float, bool]:
        return self.center, self.width, self.theta, self.p, self.keep_inside

    def calc_constraints(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position: agent position in world coordinate. shape=(2,)

        Note:
            Division by width is a remnant of the transformation.
        """
        agent_position_transformed = self._transform_agent_position(agent_position.flatten())
        sign = 1 if self.keep_inside else -1

        self.G = (
            rotation_matrix_2d(self.theta) @ self._calc_dhdx(agent_position_transformed, sign)
            / self.width.reshape(2, 1)
        ).flatten()
        assert self.G.shape == (2,)
        self.h = self._calc_h(agent_position_transformed, sign)

    def _transform_agent_position(self, agent_position: NDArray) -> NDArray:
        """Transform agent position from world coordinate to normalized area coordinate.

        Args:
            agent_position: agent position in world coordinate. shape=(2,)

        Returns:
            Transformed agent position within unit shape. shape=(2,)
        """
        return cast(NDArray, rotation_matrix_2d(-self.theta) @ (agent_position - self.center) / self.width)


class UnicyclePnorm2dCBF(Pnorm2dCBF):
    """CBF for p-norm shaped area constraint with unicycle model.

    Attributes:
        center: center of area in world coordinate. shape=(2,)
        width: half length of the major and minor axis of ellipse
            that match the x and y axis in the area coordinate. shape=(2,)
        theta: rotation angle (rad) from world to area coordinate
        p: multiplier for p-norm
        keep_inside: flag to prohibit going outside of the area
        G: constraint matrix (=dh/dx). shape=(2,)
        h: constraint value (=h(x))
    """

    def __init__(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        self.x = Matrix(symbols("x, y, agent_theta", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore
        self.set_parameters(center, width, theta, p, keep_inside)

    def set_parameters(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.center = center.flatten()
        self.width = width.flatten()
        self.theta = theta
        assert p >= 1
        self.p = p
        self.keep_inside = keep_inside

        cbf = self.sign * (
            1.0
            - sum(abs((np.hstack([np.eye(2), np.zeros([2, 1])]) @ self.x).applyfunc(lambda x: x**self.p)))
            ** (1 / self.p)
        )
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def calc_constraints(self, agent_pose: NDArray) -> None:
        """
        Args:
            agent_pose: agent pose in world coordinate. shape=(3,) [x, y, theta]

        Note:
            Division by width is a remnant of the transformation.
        """
        agent_pose_transformed = self._transform_agent_pose(agent_pose.flatten())
        sign = 1 if self.keep_inside else -1

        agent_theta = agent_pose[2]
        A = np.array(
            [
                [np.cos(agent_theta), 0],
                [np.sin(agent_theta), 0],
                [0, 1],
            ]
        )
        dhdx = self._calc_dhdx(agent_pose_transformed, sign)
        assert dhdx.shape == (3, 1), dhdx
        self.G = (
            np.vstack([rotation_matrix_2d(self.theta) @ dhdx[0:2] / self.width.reshape(2, 1), dhdx[2]]).reshape(1, -1)
            @ A
        ).flatten()
        assert self.G.shape == (2,)
        self.h = self._calc_h(agent_pose_transformed, sign)

    def _transform_agent_pose(self, agent_pose: NDArray) -> NDArray:
        """Transform agent pose from world coordinate to normalized area coordinate.

        Args:
            agent_pose: agent pose in world coordinate. shape=(3,)

        Returns:
            Transformed agent pose within unit shape. shape=(3,)
        """
        agent_position = agent_pose[0:2]
        return cast(
            NDArray,
            np.append(rotation_matrix_2d(-self.theta) @ (agent_position - self.center) / self.width, agent_pose[2]),
        )


class LiDARCBF(CBFBase):
    """CBF for LiDAR-based obstacle avoidance.

    Attributes:
        width: half length of the major and minor axis of ellipse
            that match the x and y axis in the agent coordinate. shape=(2,)
        keep_upper: flag to prohibit going lower of the limit
        G: constraint matrix (=dh/dx). shape=(2,)
        h: constraint value (=h(x))
    """

    def __init__(self, width: NDArray, keep_upper: bool = True) -> None:
        self.r = Symbol("r", real=True)  # type: ignore
        self.theta = Symbol("theta", real=True)  # type: ignore
        self.set_parameters(width, keep_upper)

    def set_parameters(self, width: NDArray, keep_upper: bool = True) -> None:
        """Set parameters and rebuild symbolic functions."""
        self.width = width.flatten()
        self.keep_upper = keep_upper

        r_c = sqrt(sum((self.width * np.array([cos(self.theta), sin(self.theta)])) ** 2))  # type: ignore
        cbf = self.r - r_c
        self._calc_dhdx = lambdify(
            [self.r, self.theta],
            [cbf.diff(self.r), cbf.diff(self.theta)],
        )
        self._calc_h = lambdify([self.r, self.theta], cbf)

    def get_parameters(self) -> Tuple[NDArray, bool]:
        return self.width, self.keep_upper

    def calc_constraints(self, r: float, theta: float) -> None:
        g_p = np.array(
            [
                [-np.cos(theta), 0],
                [np.sin(theta) / r, -1],
            ]
        )
        self.G = self._calc_dhdx(r, theta) @ g_p
        assert self.G.shape == (2,)
        self.h = self._calc_h(r, theta)

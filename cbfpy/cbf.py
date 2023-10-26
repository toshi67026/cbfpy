#!/usr/bin/env python

from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sympy import Matrix, Symbol, cos, lambdify, sin, sqrt, symbols


@dataclass
class CBFBase:
    """CBF base class

    Attributes:
        G (NDArray): constraint matrix(=dh/dx)
        h (float): constraint value(=h(x))

    Note:
        The CBF optimization problem is formulated as:
            minimize_{u} {objective function}
            subject to G*u + alpha(h) >= 0
    """

    G: NDArray
    h: float

    def get_constraints(self) -> Tuple[NDArray, float]:
        """
        Returns:
            (NDArray): G
            (NDArray): alpha(h)

        Note:
            Call get_constraints individually to apply alpha function.
        """
        return self.G, self._alpha(self.h)

    def _alpha(self, h: float) -> float:
        """
        Args:
            h (float): constraint value(=h(x))

        Returns:
            (float): h

        Note:
            If you use specific alpha function, implement it with override.
        """
        return h


class GeneralCBF(CBFBase):
    """
    Attributes:
        G (NDArray): constraint matrix(=dh/dx). shape=(1,)
        h (float): constraint value(=h(x))
    """

    def calc_constraints(self, G: NDArray, h: float) -> None:
        self.G = G
        self.h = h


class ScalarCBF(CBFBase):
    """
    Attributes:
        limit (float): limit for scalar state variable
        keep_upper (bool): flag to prohibit going lower of the limit. Defaults to True.
        x (Symbol): scalar state variable in symbolic form for cbf
        sign (Symbol): symbolic variable for cbf
        G (NDArray): constraint matrix(=dh/dx). shape=(1,)
        h (NDArray): constraint value(=h(x)). shape=(1,)

    Note:
        If keep_upper is True, the safety set is x - limit >= 0.
    """

    def __init__(self) -> None:
        self.x = Symbol("x", real=True)  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore

    def set_parameters(self, limit: float, keep_upper: bool = True) -> None:
        """Set parameters and auxiliary functions for constraint calculation"""
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
    """
    Attributes:
        a (float): lower limit for scalar state variable
        b (float): upper limit for scalar state variable
        keep_inside (bool): flag to prohibit going outside of the range. Defaults to True.
        x (Symbol): scalar state variable in symbolic form for cbf
        sign (Symbol): symbolic variable for cbf
        G (NDArray): constraint matrix(=dh/dx). shape=(1,)
        h (float): constraint value(=h(x))

    Note:
        If keep_inside is True, the safety set is a <= x <= b.
    """

    def __init__(self) -> None:
        self.x = Symbol("x", real=True)  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore

    def set_parameters(self, a: float, b: float, keep_inside: bool = True) -> None:
        """Set parameters and auxiliary functions for constraint calculation"""
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
    """
    Atrributes:
        center (NDArray): center of circular area in world coordinate. shape=(2,)
        radius (float): radius of the circular area.
        keep_inside (bool): flag to prohibit going outside of the area. Defaults to True.
        x (Matrix): state variables in symbolic form for cbf
        sign (Symbol): symbolic variable for cbf
        G (NDArray): constraint matrix(=dh/dx). shape=(2,)
        h (float): constraint value(=h(x))
    """

    def __init__(self) -> None:
        self.x = Matrix(symbols("x, y", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore

    def set_parameters(self, center: NDArray, radius: float, keep_inside: bool = True) -> None:
        """Set parameters and auxiliary functions for constraint calculation"""
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
            agent_position (NDArray): agent position in world coordinate. shape=(2,)

        Note:
            division by radius is a remnant of the transformation
        """
        agent_position_transformed = self._transform_agent_position(agent_position.flatten())
        sign = 1 if self.keep_inside else -1

        self.G = self._calc_dhdx(agent_position_transformed, sign) / self.radius
        assert self.G.shape == (2, 1)
        self.h = self._calc_h(agent_position_transformed, sign)

    def _transform_agent_position(self, agent_position: NDArray) -> NDArray:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)

        Returns:
            (NDArray): transformed agent position within unit circle. shape=(2,)
        """
        return cast(NDArray, ((agent_position - self.center) / self.radius))


class Pnorm2dCBF(CBFBase):
    """
    Atrributes:
        center (NDArray): center of area in world coordinate. shape=(2,)
        width (NDArray): half length of the major and minor axis of ellipse
            that match the x and y axis in the area coordinate. shape=(2,)
        theta (float): rotation angle(rad) world to the area coordinate.
        p (float): multiplier for p-norm.
        keep_inside (bool): flag to prohibit going outside of the area. Defaults to True.
        x (Matrix): state variables in symbolic form for cbf
        sign (Symbol): symbolic variable for cbf
        G (NDArray): constraint matrix(=dh/dx). shape=(2,)
        h (float): constraint value(=h(x))
    """

    def __init__(self) -> None:
        self.x = Matrix(symbols("x, y", real=True))  # type: ignore
        self.sign = Symbol("sign_", real=True)  # type: ignore

    def set_parameters(
        self, center: NDArray, width: NDArray, theta: float = 0.0, p: float = 2.0, keep_inside: bool = True
    ) -> None:
        """Set parameters and auxiliary functions for constraint calculation"""
        self.center = center.flatten()
        self.width = width.flatten()
        self.theta = theta
        assert p >= 1
        self.p = p
        self.keep_inside = keep_inside

        # applyfunc(lambda x: x**self.p): element-wise power
        cbf = self.sign * (1.0 - sum(abs(self.x.applyfunc(lambda x: x**self.p))) ** (1 / self.p))  # type: ignore
        self._calc_dhdx = lambdify([self.x, self.sign], cbf.diff(self.x))
        self._calc_h = lambdify([self.x, self.sign], cbf)

    def get_parameters(self) -> Tuple[NDArray, NDArray, float, float, bool]:
        """
        Returns:
            Tuple[NDArray, NDArray, float, float, bool]: parameters
        """
        return self.center, self.width, self.theta, self.p, self.keep_inside

    def calc_constraints(self, agent_position: NDArray) -> None:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)

        Note:
            division by width is a remnant of the transformation
        """
        agent_position_transformed = self._transform_agent_position(agent_position.flatten())
        sign = 1 if self.keep_inside else -1
        rotation_matrix = self._get_rotation_matrix(self.theta)

        self.G = rotation_matrix @ self._calc_dhdx(agent_position_transformed, sign) / self.width.reshape(2, 1)
        assert self.G.shape == (2, 1)
        self.h = self._calc_h(agent_position_transformed, sign)

    def _transform_agent_position(self, agent_position: NDArray) -> NDArray:
        """
        Args:
            agent_position (NDArray): agent position in world coordinate. shape=(2,)

        Returns:
            (NDArray): transformed agent position within unit circle. shape=(2,)
        """
        rotation_matrix = self._get_rotation_matrix(-self.theta)
        return cast(NDArray, rotation_matrix @ (agent_position - self.center) / self.width)

    @staticmethod
    def _get_rotation_matrix(rad: float) -> NDArray:
        return np.array(
            [
                [np.cos(rad), -np.sin(rad)],
                [np.sin(rad), np.cos(rad)],
            ]
        )


class LiDARCBF(CBFBase):
    """
    Attributes:
        width (NDArray): half length of the major and minor axis of ellipse
            that match the x and y axis in the agent coordinate. shape=(2,)
        keep_upper (bool): flag to prohibit going lower of the limit. Defaults to True.
        r: state variable in symbolic form for cbf
        theta: state variable in symbolic form for cbf
        G (NDArray): constraint matrix(=dh/dx). shape=(2,)
        h (float): constraint value(=h(x))
    """

    def __init__(self) -> None:
        self.r = Symbol("r", real=True)  # type: ignore
        self.theta = Symbol("theta", real=True)  # type: ignore

    def set_parameters(self, width: NDArray, keep_upper: bool = True) -> None:
        """Set parameters and auxiliary functions for constraint calculation"""
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
        self.h = self._calc_h(r, theta)

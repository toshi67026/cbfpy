#!/usr/bin/env python

import numpy as np
import pytest

from cbfpy.cbf import CBFBase, CircleCBF, GeneralCBF, LiDARCBF, Pnorm2dCBF, ScalarCBF, ScalarRangeCBF


class TestCBFBase:
    G = np.ones(2)
    h = 1.0
    cbf_base = CBFBase(G=G, h=h)

    def test_get_constraints(self) -> None:
        G, alpha_h = self.cbf_base.get_constraints()
        assert np.allclose(G, self.G)
        assert np.allclose(alpha_h, self.cbf_base._alpha(self.h))


class TestGeneralCBF:
    general_cbf = GeneralCBF(G=np.ones(2), h=1.0)

    def test_calc_constraints(self) -> None:
        G = 2 * np.ones(2)
        h = 2.0
        self.general_cbf.calc_constraints(G, h)
        ret_G, ret_alpha_h = self.general_cbf.get_constraints()
        assert np.allclose(ret_G, G)
        assert np.allclose(ret_alpha_h, self.general_cbf._alpha(h))


class TestScalarCBF:
    limit = 1.0
    keep_upper = True

    def test_set_and_get_parameters(self) -> None:
        scalar_cbf = ScalarCBF()
        scalar_cbf.set_parameters(self.limit, self.keep_upper)
        limit, keep_upper = scalar_cbf.get_parameters()
        assert limit == pytest.approx(self.limit)
        assert keep_upper == self.keep_upper

    def test_calc_constraints(self) -> None:
        scalar_cbf = ScalarCBF()
        scalar_cbf.set_parameters(self.limit, self.keep_upper)
        scalar_cbf.calc_constraints(3.0)
        G, alpha_h = scalar_cbf.get_constraints()
        assert np.allclose(G, np.array(1.0))
        assert alpha_h == pytest.approx(2.0)


class TestScalarRangeCBF:
    a = 1.0
    b = 4.0
    keep_inside = True

    def test_set_and_get_parameters(self) -> None:
        scalar_range_cbf = ScalarRangeCBF()
        scalar_range_cbf.set_parameters(self.a, self.b, self.keep_inside)
        a, b, keep_inside = scalar_range_cbf.get_parameters()
        assert a == pytest.approx(self.a)
        assert b == pytest.approx(self.b)
        assert keep_inside == self.keep_inside

    def test_calc_constraints(self) -> None:
        scalar_range_cbf = ScalarRangeCBF()
        scalar_range_cbf.set_parameters(self.a, self.b, self.keep_inside)

        scalar_range_cbf.calc_constraints(2.0)
        G, alpha_h = scalar_range_cbf.get_constraints()
        assert np.allclose(G, np.array(1.0))
        assert alpha_h == pytest.approx(2.0)


class TestCircleCBF:
    center = np.ones(2)
    radius = 2.0
    keep_inside = True

    def test_set_and_get_parameters(self) -> None:
        circle_cbf = CircleCBF()
        circle_cbf.set_parameters(self.center, self.radius, self.keep_inside)
        center, radius, keep_inside = circle_cbf.get_parameters()
        assert np.allclose(center, self.center)
        assert radius == pytest.approx(self.radius)
        assert keep_inside == self.keep_inside

    def test_calc_constraints(self) -> None:
        circle_cbf = CircleCBF()
        circle_cbf.set_parameters(self.center, self.radius, self.keep_inside)

        agent_position = 2 * np.ones(2)
        circle_cbf.calc_constraints(agent_position)
        G, alpha_h = circle_cbf.get_constraints()
        assert np.allclose(G, np.array([-0.35355339, -0.35355339]))
        assert alpha_h == pytest.approx(0.2928932188134524)


class TestPnorm2dCBF:
    center = np.ones(2)
    width = np.array([2, 1])
    theta = 0.3
    p = 4.0
    keep_inside = True

    def test_set_and_get_parameters(self) -> None:
        pnorm2d_cbf = Pnorm2dCBF()
        pnorm2d_cbf.set_parameters(self.center, self.width, self.theta, self.p, self.keep_inside)
        center, width, theta, p, keep_inside = pnorm2d_cbf.get_parameters()
        assert np.allclose(center, self.center)
        assert np.allclose(width, self.width)
        assert theta == pytest.approx(self.theta)
        assert p == pytest.approx(self.p)
        assert keep_inside == self.keep_inside

    def test_calc_constraints(self) -> None:
        pnorm2d_cbf = Pnorm2dCBF()
        pnorm2d_cbf.set_parameters(self.center, self.width, self.theta, self.p, self.keep_inside)

        agent_position = 2 * np.ones(2)
        pnorm2d_cbf.calc_constraints(agent_position)
        G, alpha_h = pnorm2d_cbf.get_constraints()
        assert np.allclose(G, np.array([-0.1661934, -0.77436655]))
        assert alpha_h == pytest.approx(0.234969565)


class TestLiDARCBF:
    width = np.array([2, 1])
    keep_upper = True

    def test_set_and_get_parameters(self) -> None:
        lidar_cbf = LiDARCBF()
        lidar_cbf.set_parameters(self.width, self.keep_upper)
        width, keep_upper = lidar_cbf.get_parameters()
        assert np.allclose(width, self.width)
        assert keep_upper == self.keep_upper

    def test_calc_constraints(self) -> None:
        lidar_cbf = LiDARCBF()
        lidar_cbf.set_parameters(self.width, self.keep_upper)

        lidar_cbf.calc_constraints(3.0, 0.0)
        G, alpha_h = lidar_cbf.get_constraints()
        assert np.allclose(G, np.array([-1.0, 0.0]))
        assert alpha_h == pytest.approx(1.0)

"""分时段运行情景的测试（unittest，可直接 python -m unittest 运行）。"""

import unittest

import numpy as np

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.farm.operation import (
    MONTH_HOURS_NON_LEAP,
    OperatingPeriod,
    OperatingScenarioSet,
    ScenarioEnergyCalculator,
    allocate_curtailment,
    build_monthly_scenario,
    build_scenario_from_dicts,
    scale_wind_resource,
)
from wind_farm_opt.optimization.baseline import generate_grid_layout


class ScenarioTestBase(unittest.TestCase):
    n = 6

    def setUp(self) -> None:
        self.turbines = [
            create_default_turbine("V126-3.45MW") for _ in range(self.n)
        ]
        self.wr = create_default_wind_resource(12, 270.0, 8.5)
        boundary = create_rectangular_boundary(3500, 3500)
        self.positions = generate_grid_layout(
            boundary,
            self.n,
            np.array([t.rotor_diameter for t in self.turbines]),
            rng=np.random.default_rng(0),
        )
        self.calc = AEPCalculator(
            self.turbines, self.wr, JensenWake(0.07), speed_step=1.0
        )
        self.scenario_calc = ScenarioEnergyCalculator(self.calc)
        self.installed_mw = self.n * 3.45


class TestLegacyEquivalence(ScenarioTestBase):
    """未配置/全可用无限电网的情景必须与旧 AEP 流程数值一致。"""

    def test_monthly_all_available_matches_legacy(self) -> None:
        farm = self.calc.compute_farm_aep(self.positions)
        scenario = build_monthly_scenario(self.wr, self.n, availability=1.0)
        result = self.scenario_calc.compute(self.positions, scenario)

        self.assertAlmostEqual(result.gross_mwh, farm.gross_aep, places=6)
        self.assertAlmostEqual(result.wake_loss_mwh, farm.total_wake_loss, places=6)
        # 无不可利用、无限发：上网电量 == 旧净 AEP
        self.assertAlmostEqual(result.delivered_mwh, farm.net_aep, places=6)
        self.assertEqual(result.total_hours, 8760.0)

    def test_no_double_counting_hours_split(self) -> None:
        """把全年拆成两个等权时段，电量应与单时段一致（小时数不重复计入）。"""
        half1 = [
            OperatingPeriod(f"H1-{m}", MONTH_HOURS_NON_LEAP[m], self.wr, np.ones(self.n))
            for m in range(6)
        ]
        half2 = [
            OperatingPeriod(f"H2-{m}", MONTH_HOURS_NON_LEAP[m + 6], self.wr, np.ones(self.n))
            for m in range(6)
        ]
        split = self.scenario_calc.compute(
            self.positions, OperatingScenarioSet(half1 + half2)
        )
        full = self.scenario_calc.compute(
            self.positions, build_monthly_scenario(self.wr, self.n)
        )
        self.assertAlmostEqual(split.gross_mwh, full.gross_mwh, places=6)
        self.assertAlmostEqual(split.delivered_mwh, full.delivered_mwh, places=6)

    def test_weighted_representative_periods(self) -> None:
        """代表性时段的 weight 只放大已积分电量，hours 不再重复乘。"""
        periods = [
            OperatingPeriod(
                "rep", 876.0, self.wr, np.ones(self.n), weight=10.0
            )
        ]
        result = self.scenario_calc.compute(
            self.positions, OperatingScenarioSet(periods)
        )
        single = self.scenario_calc.compute_period(
            self.positions, periods[0]
        )
        self.assertEqual(result.total_hours, 8760.0)
        self.assertAlmostEqual(result.gross_mwh, single.gross * 10.0, places=6)


class TestEnergyTiers(ScenarioTestBase):
    def test_five_tier_identity(self) -> None:
        avail = np.ones((12, self.n))
        avail[0, 0] = 0.5
        avail[1, :] = 0.8
        caps = [None] * 12
        caps[6] = 3.0
        scenario = build_monthly_scenario(
            self.wr, self.n, availability=avail, grid_capacity_mw=caps
        )
        r = self.scenario_calc.compute(self.positions, scenario)

        residual = (
            r.gross_mwh
            - r.wake_loss_mwh
            - r.availability_loss_mwh
            - r.curtailment_loss_mwh
            - r.delivered_mwh
        )
        self.assertAlmostEqual(residual, 0.0, places=6)
        # 各项损失非负，上网电量为正
        self.assertGreaterEqual(r.wake_loss_mwh, 0.0)
        self.assertGreater(r.availability_loss_mwh, 0.0)
        self.assertGreater(r.curtailment_loss_mwh, 0.0)
        self.assertGreater(r.delivered_mwh, 0.0)
        # 百分比以毛发电为分母并闭合
        d = r.as_dict()
        self.assertAlmostEqual(
            d["wake_loss_pct"]
            + d["availability_loss_pct"]
            + d["curtailment_loss_pct"]
            + d["delivered_pct"],
            100.0,
            places=6,
        )

    def test_availability_reduces_delivered(self) -> None:
        full = self.scenario_calc.compute(
            self.positions, build_monthly_scenario(self.wr, self.n, availability=1.0)
        )
        half = self.scenario_calc.compute(
            self.positions, build_monthly_scenario(self.wr, self.n, availability=0.5)
        )
        # 均匀 0.5 可利用率：可用与上网电量约为全可用的一半，毛发电与尾流不变
        self.assertAlmostEqual(full.gross_mwh, half.gross_mwh, places=6)
        self.assertAlmostEqual(
            half.delivered_mwh, full.delivered_mwh * 0.5, places=4
        )

    def test_curtailment_capacity_energy_ceiling(self) -> None:
        """限发后该时段电量不超过 cap*hours，且削减量等于超限电量。"""
        caps = [None] * 12
        caps[6] = 3.0
        r = self.scenario_calc.compute(
            self.positions,
            build_monthly_scenario(self.wr, self.n, grid_capacity_mw=caps),
        )
        jul = next(p for p in r.period_results if p.name == "7月")
        self.assertAlmostEqual(jul.delivered, 3.0 * jul.hours, places=6)
        self.assertAlmostEqual(
            jul.curtailment_loss,
            jul.gross - jul.wake_loss - jul.delivered,
            places=6,
        )


class TestCurtailmentAllocation(unittest.TestCase):
    def test_proportional(self) -> None:
        power = np.array([4.0, 2.0])  # MW, total 6 > cap 3
        curt = allocate_curtailment(power, hours=10.0, grid_capacity_mw=3.0)
        # 超限 3 MW * 10h = 30 MWh，按 2:1 分摊
        np.testing.assert_allclose(curt, [20.0, 10.0], rtol=1e-12)

    def test_merit_order(self) -> None:
        power = np.array([4.0, 2.0])  # 优先保 0 号机
        curt = allocate_curtailment(
            power,
            hours=10.0,
            grid_capacity_mw=3.0,
            strategy="merit_order",
            merit_order=[0, 1],
        )
        # 0 号机保 4 MW 中的 3 MW 容量？总容量3：先保0号 -> 0号发3MW(限1MW)，1号全限
        np.testing.assert_allclose(curt, [10.0, 20.0], rtol=1e-12)

    def test_no_curtailment_when_under_cap(self) -> None:
        power = np.array([1.0, 1.0])
        curt = allocate_curtailment(power, 5.0, 10.0)
        np.testing.assert_array_equal(curt, np.zeros(2))
        curt_none = allocate_curtailment(power, 5.0, None)
        np.testing.assert_array_equal(curt_none, np.zeros(2))


class TestValidationRejection(ScenarioTestBase):
    def _expect(self, periods, **kw) -> None:
        with self.assertRaises(ValueError):
            OperatingScenarioSet(periods).validate(
                self.n, installed_capacity_mw=self.installed_mw, **kw
            )

    def test_empty_periods(self) -> None:
        self._expect([])

    def test_missing_month_not_closed(self) -> None:
        # 只有 11 个月（权重 1），小时数不闭合到 8760
        periods = [
            OperatingPeriod(f"m{i}", 730.0, self.wr, np.ones(self.n))
            for i in range(11)
        ]
        self._expect(periods)

    def test_weight_not_closed(self) -> None:
        periods = [
            OperatingPeriod("a", 4380.0, self.wr, np.ones(self.n), weight=1.0),
            OperatingPeriod("b", 4380.0, self.wr, np.ones(self.n), weight=0.9),
        ]
        self._expect(periods)

    def test_negative_availability(self) -> None:
        av = np.ones(self.n)
        av[2] = -0.1
        self._expect([OperatingPeriod("x", 8760.0, self.wr, av)])

    def test_availability_above_one(self) -> None:
        av = np.ones(self.n)
        av[2] = 1.05
        self._expect([OperatingPeriod("x", 8760.0, self.wr, av)])

    def test_availability_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            OperatingPeriod(
                "x", 8760.0, self.wr, np.ones(self.n + 1)
            ).validate(self.n)

    def test_capacity_conflict_above_installed(self) -> None:
        self._expect(
            [
                OperatingPeriod(
                    "x", 8760.0, self.wr, np.ones(self.n),
                    grid_capacity_mw=self.installed_mw + 10.0,
                )
            ]
        )

    def test_zero_or_negative_hours(self) -> None:
        self._expect([OperatingPeriod("x", 0.0, self.wr, np.ones(self.n))])

    def test_nonpositive_weight(self) -> None:
        self._expect(
            [
                OperatingPeriod(
                    "x", 876.0, self.wr, np.ones(self.n), weight=0.0
                )
            ]
        )

    def test_duplicate_names(self) -> None:
        self._expect(
            [
                OperatingPeriod("a", 4380.0, self.wr, np.ones(self.n)),
                OperatingPeriod("a", 4380.0, self.wr, np.ones(self.n)),
            ]
        )

    def test_bad_merit_order(self) -> None:
        with self.assertRaises(ValueError):
            OperatingPeriod(
                "x", 8760.0, self.wr, np.ones(self.n),
                curtailment_strategy="merit_order",
                merit_order=[0, 0, 1, 2, 3, 4],
            ).validate(self.n)

    def test_bad_strategy(self) -> None:
        with self.assertRaises(ValueError):
            OperatingPeriod(
                "x", 8760.0, self.wr, np.ones(self.n),
                curtailment_strategy="unknown",
            ).validate(self.n)

    def test_calculator_rejects_before_running(self) -> None:
        """非法情景必须在任何电量计算前抛出（计算器入口）。"""
        bad = build_monthly_scenario(self.wr, self.n, availability=-0.2)
        with self.assertRaises(ValueError):
            self.scenario_calc.compute(self.positions, bad)


class TestWindResourceScaling(unittest.TestCase):
    def test_scale_preserves_frequency_and_k(self) -> None:
        wr = create_default_wind_resource(12, 270.0, 8.5)
        scaled = scale_wind_resource(wr, 1.2)
        for s0, s1 in zip(wr.sectors, scaled.sectors):
            self.assertAlmostEqual(s0.frequency, s1.frequency)
            self.assertAlmostEqual(s0.weibull_k, s1.weibull_k)
            self.assertAlmostEqual(s1.mean_speed, s0.mean_speed * 1.2)
            self.assertAlmostEqual(s1.weibull_c, s0.weibull_c * 1.2)

    def test_nonpositive_factor_rejected(self) -> None:
        wr = create_default_wind_resource(12, 270.0, 8.5)
        with self.assertRaises(ValueError):
            scale_wind_resource(wr, 0.0)


class TestCustomDictBuilder(ScenarioTestBase):
    def test_custom_periods_build_and_compute(self) -> None:
        dicts = [
            {
                "name": "winter",
                "hours": 2160,
                "wind_resource": {"speed_factor": 1.15},
                "availability": [0.7] + [1.0] * (self.n - 1),
                "grid_capacity_mw": 15.0,
            },
            {"name": "rest", "hours": 6600, "availability": 1.0},
        ]
        scenario = build_scenario_from_dicts(dicts, self.n, self.wr)
        result = self.scenario_calc.compute(self.positions, scenario)
        self.assertEqual(result.total_hours, 8760.0)
        self.assertGreater(result.availability_loss_mwh, 0.0)

    def test_missing_hours_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario_from_dicts(
                [{"name": "x", "availability": 1.0}], self.n, self.wr
            )

    def test_bad_availability_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_scenario_from_dicts(
                [{"name": "x", "hours": 8760, "availability": [1.0, 1.0]}],
                self.n,
                self.wr,
            )


if __name__ == "__main__":
    unittest.main()

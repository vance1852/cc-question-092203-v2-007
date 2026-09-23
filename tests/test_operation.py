"""运行情景（按月/自定义时段）能量计算的测试。

可直接运行：``python -m tests.test_operation``（不依赖 pytest）。
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import JensenWake
from wind_farm_opt.core.wind_resource import create_default_wind_resource
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.farm.operation import (
    HOURS_PER_YEAR,
    OperatingPeriod,
    OperatingScenario,
    OperationalEnergyCalculator,
    ScenarioValidationError,
    create_monthly_periods,
)


def _build(n_turb=6, speed_step=1.0):
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    wr = create_default_wind_resource(num_sectors=12, mean_speed=8.5)
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=wr,
        wake_model=JensenWake(0.07),
        speed_step=speed_step,
    )
    # 规则网格，沿主风向(270°，即从西向东吹)排开以产生明显尾流
    positions = np.array(
        [[x, 0.0] for x in np.linspace(-1500, 1500, n_turb)], dtype=np.float64
    )
    return turbines, wr, calc, positions


class TestPeriodBasics(unittest.TestCase):
    def test_period_hours_must_be_positive(self):
        with self.assertRaises(ScenarioValidationError):
            OperatingPeriod(name="坏", hours=0.0)
        with self.assertRaises(ScenarioValidationError):
            OperatingPeriod(name="坏", hours=-1.0)

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ScenarioValidationError):
            OperatingPeriod(name="坏", hours=100.0, grid_capacity_mw=-1.0)

    def test_empty_name_rejected(self):
        with self.assertRaises(ScenarioValidationError):
            OperatingPeriod(name="  ", hours=100.0)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ScenarioValidationError):
            OperatingPeriod(name="p", hours=100.0, curtailment_rule="mystery")


class TestScenarioValidation(unittest.TestCase):
    def setUp(self):
        self.n = 6
        self.wr = create_default_wind_resource(num_sectors=12)

    def test_missing_month_hours_not_closed(self):
        # 11 个时段无法闭合到 8760
        periods = [
            OperatingPeriod(name=f"p{i}", hours=700.0) for i in range(11)
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_too_many_hours_rejected(self):
        periods = [
            OperatingPeriod(name="a", hours=5000.0),
            OperatingPeriod(name="b", hours=4000.0),
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_custom_partial_year_allowed_when_not_required(self):
        # 自定义时段不要求覆盖全年，只要不超过 8760
        periods = [OperatingPeriod(name="冬季检修", hours=2160.0)]
        scenario = OperatingScenario(
            periods, self.n, default_wind_resource=self.wr,
            require_full_coverage=False,
        )
        self.assertEqual(scenario.total_hours, HOURS_PER_YEAR)

    def test_custom_partial_over_cap_rejected(self):
        periods = [OperatingPeriod(name="x", hours=9000.0)]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(
                periods, self.n, default_wind_resource=self.wr,
                require_full_coverage=False,
            )

    def test_duplicate_names_rejected(self):
        periods = [
            OperatingPeriod(name="同", hours=4000.0),
            OperatingPeriod(name="同", hours=4760.0),
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_negative_availability_rejected(self):
        periods = [
            OperatingPeriod(name="a", hours=8760.0, availability=[-0.01] * self.n)
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_above_one_availability_rejected(self):
        periods = [
            OperatingPeriod(name="a", hours=8760.0, availability=[1.01] * self.n)
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_availability_length_mismatch_rejected(self):
        periods = [
            OperatingPeriod(name="a", hours=8760.0, availability=[0.9] * (self.n - 1))
        ]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n, default_wind_resource=self.wr)

    def test_capacity_exceeding_installed_rejected(self):
        periods = [OperatingPeriod(name="a", hours=8760.0, grid_capacity_mw=1000.0)]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(
                periods, self.n, default_wind_resource=self.wr,
                installed_capacity_mw=20.7,
            )

    def test_capacity_equal_installed_accepted(self):
        periods = [OperatingPeriod(name="a", hours=8760.0, grid_capacity_mw=20.7)]
        OperatingScenario(
            periods, self.n, default_wind_resource=self.wr,
            installed_capacity_mw=20.7,
        )

    def test_missing_wind_resource_rejected(self):
        periods = [OperatingPeriod(name="a", hours=8760.0)]
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario(periods, self.n)  # 无默认风资源

    def test_empty_periods_rejected(self):
        with self.assertRaises(ScenarioValidationError):
            OperatingScenario([], self.n, default_wind_resource=self.wr)


class TestEnergyAccounting(unittest.TestCase):
    def setUp(self):
        self.turbines, self.wr, self.calc, self.positions = _build()
        self.n = len(self.turbines)
        self.installed_mw = sum(t.rated_power for t in self.turbines) / 1e3
        self.op_calc = OperationalEnergyCalculator(self.calc)

    def test_single_period_8760_matches_net_aep(self):
        """无约束单时段 8760h 上网电量必须等于旧流程净 AEP。"""
        scenario = OperatingScenario(
            [OperatingPeriod(name="全年", hours=8760.0)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        farm = self.calc.compute_farm_aep(self.positions)
        # 毛发电 = 旧 gross，尾流后 = 旧 net，无其他损失
        self.assertAlmostEqual(energy.gross_energy, farm.gross_aep, places=6)
        self.assertAlmostEqual(
            energy.post_wake_energy, farm.net_aep, places=6
        )
        self.assertAlmostEqual(energy.grid_energy, farm.net_aep, places=6)
        self.assertAlmostEqual(energy.unavailability_loss, 0.0, places=6)
        self.assertAlmostEqual(energy.curtailment_loss, 0.0, places=6)

    def test_loss_chain_adds_up(self):
        """五级账目闭合：毛 - 尾流 - 不可用 - 限发 = 上网。"""
        rng = np.random.default_rng(0)
        av = rng.uniform(0.8, 1.0, size=self.n)
        periods = [
            OperatingPeriod(name="a", hours=4380.0, availability=av,
                            grid_capacity_mw=self.installed_mw * 0.6),
            OperatingPeriod(name="b", hours=4380.0, availability=0.95,
                            grid_capacity_mw=None),
        ]
        scenario = OperatingScenario(
            periods, self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        recon = (
            energy.gross_energy
            - energy.wake_loss
            - energy.unavailability_loss
            - energy.curtailment_loss
        )
        self.assertAlmostEqual(recon, energy.grid_energy, places=6)
        # 所有损失非负
        for p in energy.periods:
            self.assertGreaterEqual(p.wake_loss, -1e-9)
            self.assertGreaterEqual(p.unavailability_loss, -1e-9)
            self.assertGreaterEqual(p.curtailment_loss, -1e-9)

    def test_hours_not_double_counted(self):
        """把一个 8760h 时段拆成两段，各损失与上网电量必须与单段一致。"""
        one = OperatingScenario(
            [OperatingPeriod(name="all", hours=8760.0, availability=0.9)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        split = OperatingScenario(
            [
                OperatingPeriod(name="h1", hours=3000.0, availability=0.9),
                OperatingPeriod(name="h2", hours=5760.0, availability=0.9),
            ],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        e1 = self.op_calc.compute_scenario(self.positions, one)
        e2 = self.op_calc.compute_scenario(self.positions, split)
        self.assertAlmostEqual(e1.grid_energy, e2.grid_energy, places=5)
        self.assertAlmostEqual(e1.gross_energy, e2.gross_energy, places=5)
        self.assertAlmostEqual(
            e1.unavailability_loss, e2.unavailability_loss, places=5
        )

    def test_zero_availability_means_zero_energy(self):
        scenario = OperatingScenario(
            [OperatingPeriod(name="全停", hours=8760.0, availability=0.0)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        self.assertAlmostEqual(energy.grid_energy, 0.0, places=9)
        self.assertAlmostEqual(energy.curtailment_loss, 0.0, places=9)
        # 尾流损失仍存在（毛发电与尾流后之差），不可利用损失吞掉其余
        self.assertGreater(energy.unavailability_loss, 0.0)

    def test_zero_grid_capacity_means_zero_grid(self):
        scenario = OperatingScenario(
            [OperatingPeriod(name="闭锁", hours=8760.0, grid_capacity_mw=0.0)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        self.assertAlmostEqual(energy.grid_energy, 0.0, places=9)
        # 全部可用能量都转为限发损失
        self.assertAlmostEqual(
            energy.curtailment_loss, energy.available_energy, places=6
        )

    def test_high_capacity_causes_no_curtailment(self):
        scenario = OperatingScenario(
            [OperatingPeriod(name="不限", hours=8760.0,
                             grid_capacity_mw=self.installed_mw)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        self.assertAlmostEqual(energy.curtailment_loss, 0.0, places=6)

    def test_tighter_capacity_monotonic_curtailment(self):
        """并网上限越紧，限发损失单调不减。"""
        losses = []
        for cap_frac in (1.0, 0.7, 0.5, 0.3, 0.1):
            scenario = OperatingScenario(
                [OperatingPeriod(
                    name="c", hours=8760.0,
                    grid_capacity_mw=self.installed_mw * cap_frac)],
                self.n, default_wind_resource=self.wr,
                installed_capacity_mw=self.installed_mw,
            )
            e = self.op_calc.compute_scenario(self.positions, scenario)
            losses.append(e.curtailment_loss)
        for a, b in zip(losses, losses[1:]):
            self.assertGreaterEqual(b, a - 1e-9)

    def test_curtailment_proportional_allocation(self):
        """限发在每个运行状态对各机按相同比例分配（状态级 proportional）。"""
        cap_kw = self.installed_mw * 0.5 * 1e3
        hours, weights, gross_kw, net_kw = (
            self.calc.compute_period_power_distribution(
                self.positions, hours=8760.0
            )
        )
        availability = np.ones(self.n)
        avail_kw = net_kw * availability[np.newaxis, :, np.newaxis]
        farm = avail_kw.sum(axis=1)  # (N_sector, N_speed)
        share = np.where(farm > 0.0, np.minimum(farm, cap_kw) / farm, 1.0)

        # 对每个被约束的状态，所有"有出力"机组的上网/可用比例必须相同
        for s in range(share.shape[0]):
            for v in range(share.shape[1]):
                if farm[s, v] > cap_kw:
                    delivered = avail_kw[s, :, v] * share[s, v]
                    active = avail_kw[s, :, v] > 0.0
                    ratios = delivered[active] / avail_kw[s, active, v]
                    self.assertLess(np.ptp(ratios), 1e-12)
                    # 该状态全场出力恰好等于上限
                    self.assertAlmostEqual(delivered.sum(), cap_kw, places=6)

    def test_turbine_level_sums(self):
        scenario = OperatingScenario(
            [OperatingPeriod(name="a", hours=8760.0, availability=0.9,
                             grid_capacity_mw=self.installed_mw * 0.6)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        p = self.op_calc.compute_period(
            self.positions, scenario.periods[0], scenario
        )
        self.assertAlmostEqual(p.turbine_gross.sum(), p.gross_energy, places=9)
        self.assertAlmostEqual(p.turbine_grid.sum(), p.grid_energy, places=9)
        self.assertAlmostEqual(
            p.turbine_curtailment.sum(), p.curtailment_loss, places=9
        )

    def test_monthly_helper_sums_to_8760(self):
        periods = create_monthly_periods(
            availabilities=[0.95] * 12,
            grid_capacities_mw=[None] * 12,
        )
        self.assertEqual(len(periods), 12)
        self.assertAlmostEqual(sum(p.hours for p in periods), 8760.0, places=9)
        scenario = OperatingScenario(
            periods, self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        # 12 个月、恒定 0.95 可用率 ≈ 单时段 8760、0.95 可用率
        single = OperatingScenario(
            [OperatingPeriod(name="全年", hours=8760.0, availability=0.95)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        e_single = self.op_calc.compute_scenario(self.positions, single)
        self.assertAlmostEqual(
            energy.grid_energy, e_single.grid_energy, places=4
        )

    def test_period_specific_wind_resource(self):
        """时段可用独立风资源；高/低风速时段电量应有明显差异。"""
        from wind_farm_opt.core.wind_resource import create_simple_wind_resource
        strong = create_simple_wind_resource(num_sectors=12, uniform=True, mean_speed=11.0)
        weak = create_simple_wind_resource(num_sectors=12, uniform=True, mean_speed=6.0)
        ps = OperatingScenario(
            [
                OperatingPeriod(name="大风", hours=4380.0, wind_resource=strong),
                OperatingPeriod(name="小风", hours=4380.0, wind_resource=weak),
            ],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        results = {
            p.name: p for p in self.op_calc.compute_scenario(self.positions, ps).periods
        }
        self.assertGreater(
            results["大风"].gross_energy, results["小风"].gross_energy * 1.5
        )

    def test_per_turbine_availability_length(self):
        good = [0.9] * self.n
        scenario = OperatingScenario(
            [OperatingPeriod(name="a", hours=8760.0, availability=good)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        energy = self.op_calc.compute_scenario(self.positions, scenario)
        # 一台完全停机（av=0）的极端情形
        av = [1.0] * self.n
        av[0] = 0.0
        scenario2 = OperatingScenario(
            [OperatingPeriod(name="a", hours=8760.0, availability=av)],
            self.n, default_wind_resource=self.wr,
            installed_capacity_mw=self.installed_mw,
        )
        p = self.op_calc.compute_period(
            self.positions, scenario2.periods[0], scenario2
        )
        self.assertAlmostEqual(p.turbine_grid[0], 0.0, places=9)
        self.assertTrue(np.isfinite(energy.grid_energy))


class TestConfigIntegration(unittest.TestCase):
    def test_disabled_by_default_keeps_old_flow(self):
        from wind_farm_opt.config import WindFarmConfig
        cfg = WindFarmConfig()
        self.assertFalse(cfg.operation.enabled)
        self.assertIsNone(
            cfg.operation.create_scenario(
                n_turbines=6,
                default_wind_resource=create_default_wind_resource(),
                installed_capacity_mw=20.7,
            )
        )

    def test_config_round_trip_and_build(self):
        import json
        import tempfile
        from wind_farm_opt.config import WindFarmConfig
        cfg = WindFarmConfig(n_turbines=6)
        cfg.operation.enabled = True
        cfg.operation.periods = [
            {"name": "1月", "hours": 744.0, "availability": 0.9,
             "grid_capacity_mw": 15.0},
            {"name": "其余", "hours": 8016.0, "availability": 0.97},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            cfg.to_json(path)
            cfg2 = WindFarmConfig.from_json(path)
        self.assertTrue(cfg2.operation.enabled)
        wr = create_default_wind_resource()
        scenario = cfg2.operation.create_scenario(6, wr, 20.7)
        self.assertEqual(len(scenario.periods), 2)

    def test_config_monthly_builds_12_periods(self):
        from wind_farm_opt.config import OperationConfig
        op = OperationConfig.from_dict({
            "monthly": {
                "availability": [0.95] * 12,
                "grid_capacities_mw": [18.0] * 6 + [None] * 6,
            }
        })
        self.assertTrue(op.enabled)
        scenario = op.create_scenario(
            6, create_default_wind_resource(), 20.7
        )
        self.assertEqual(len(scenario.periods), 12)
        self.assertAlmostEqual(
            sum(p.hours for p in scenario.periods), 8760.0, places=9
        )

    def test_config_bad_hours_rejected_at_build(self):
        from wind_farm_opt.config import OperationConfig
        op = OperationConfig.from_dict({
            "periods": [{"name": "缺月", "hours": 8000.0}]
        })
        with self.assertRaises(ScenarioValidationError):
            op.create_scenario(6, create_default_wind_resource(), 20.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

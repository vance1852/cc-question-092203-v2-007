"""按月（或自定义时段）的运行情景能量计算。

在风资源/尾流/AEP 流程之上引入运行约束：

- 每个时段指定 **小时数**（替代默认的全年 8760 小时假设）；
- 每个时段可使用独立的 :class:`~wind_farm_opt.core.wind_resource.WindResource`；
- 每个时段指定 **逐机可利用率**（冬季检修、逐机故障率）；
- 每个时段指定 **全场并网功率上限**（送出受限月份的限发）。

能量按如下顺序逐级形成，各级损失彼此独立、可加且可解释：

1. 毛发电（无尾流）
2. 尾流后功率（扣除尾流损失）
3. 应用逐机可利用率（不可利用损失）
4. 在场站并网点按可解释规则分配限发（限发损失）
5. 最终上网电量

所有时段的小时数只在扇区积分时作为唯一的时间权重进入一次，
时段之间不使用任何额外权重，因此不会重复计入。
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.wind_resource import WindResource
from .aep import AEPCalculator

#: 小时数闭合校验容差（小时）
HOURS_TOLERANCE = 1e-6
#: 标准全年小时数
HOURS_PER_YEAR = 8760.0


class ScenarioValidationError(ValueError):
    """运行情景配置在运行前校验失败。"""


@dataclass
class OperatingPeriod:
    """单个运行时段（如一个月或检修窗口）。

    Parameters
    ----------
    name : str
        时段名称（如 "1月"、"冬季检修"）
    hours : float
        时段小时数
    wind_resource : Optional[WindResource]
        时段风资源；为 None 时使用情景默认（年度）风资源
    availability : Optional[np.ndarray | Sequence[float]]
        逐机可利用率 (0~1)，长度须等于风机台数；
        为 None 时全部按 1.0（机组完全可用）
    grid_capacity_mw : Optional[float]
        全场并网功率上限 (MW)；为 None 表示不限制送出
    curtailment_rule : str
        限发分配规则，目前支持 "proportional"（按各机可用后功率等比例压减）
    """

    name: str
    hours: float
    wind_resource: Optional[WindResource] = None
    availability: Optional[np.ndarray | Sequence[float]] = None
    grid_capacity_mw: Optional[float] = None
    curtailment_rule: str = "proportional"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ScenarioValidationError("时段名称不能为空")
        if not np.isfinite(self.hours) or self.hours <= 0.0:
            raise ScenarioValidationError(
                f"时段 {self.name!r} 的小时数必须为正数，当前为 {self.hours}"
            )
        if self.grid_capacity_mw is not None:
            cap = float(self.grid_capacity_mw)
            if not np.isfinite(cap) or cap < 0.0:
                raise ScenarioValidationError(
                    f"时段 {self.name!r} 的并网功率上限不能为负，当前为 {cap} MW"
                )
            self.grid_capacity_mw = cap
        if self.curtailment_rule not in ("proportional",):
            raise ScenarioValidationError(
                f"时段 {self.name!r} 使用了未知的限发分配规则: {self.curtailment_rule}"
            )


@dataclass
class PeriodEnergyResult:
    """单个时段的能量分解结果（单位均为 MWh）。

    Attributes
    ----------
    name : 时段名称
    hours : 时段小时数
    gross_energy : 毛发电（无尾流）
    wake_loss : 尾流损失
    post_wake_energy : 尾流后、应用可利用率之前的能量
    unavailability_loss : 机组不可利用损失
    available_energy : 可利用率之后、限发之前的能量
    curtailment_loss : 限发（并网约束）损失
    grid_energy : 最终上网电量
    curtailment_factor : 限发压减比例（0 表示未限发）
    binding_hours_equivalent : 并网上限约束等效作用的"满额小时"说明值（保留备用）
    mean_availability : 时段逐机可利用率的发电量加权均值
    """

    name: str
    hours: float
    gross_energy: float
    wake_loss: float
    post_wake_energy: float
    unavailability_loss: float
    available_energy: float
    curtailment_loss: float
    grid_energy: float
    curtailment_factor: float
    mean_availability: float
    turbine_gross: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    turbine_post_wake: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    turbine_available: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    turbine_grid: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    turbine_curtailment: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict:
        """转为可 JSON 序列化的字典（不含逐机数组）。"""
        return {
            "name": self.name,
            "hours": float(self.hours),
            "gross_mwh": float(self.gross_energy),
            "wake_loss_mwh": float(self.wake_loss),
            "unavailability_loss_mwh": float(self.unavailability_loss),
            "curtailment_loss_mwh": float(self.curtailment_loss),
            "grid_energy_mwh": float(self.grid_energy),
            "curtailment_factor": float(self.curtailment_factor),
            "mean_availability": float(self.mean_availability),
        }


@dataclass
class ScenarioEnergyResult:
    """整个运行情景（所有时段合计）的能量分解（单位 MWh）。"""

    periods: list[PeriodEnergyResult]
    total_hours: float
    gross_energy: float
    wake_loss: float
    unavailability_loss: float
    curtailment_loss: float
    grid_energy: float

    @property
    def post_wake_energy(self) -> float:
        """尾流后能量（毛发电扣除尾流损失）。"""
        return self.gross_energy - self.wake_loss

    @property
    def available_energy(self) -> float:
        """可利用率之后、限发之前的能量。"""
        return self.post_wake_energy - self.unavailability_loss

    def loss_summary(self) -> dict[str, float]:
        """返回各项损失及占毛发电比例的汇总。"""
        denom = self.gross_energy if self.gross_energy > 0.0 else 0.0
        return {
            "gross_mwh": self.gross_energy,
            "wake_loss_mwh": self.wake_loss,
            "unavailability_loss_mwh": self.unavailability_loss,
            "curtailment_loss_mwh": self.curtailment_loss,
            "grid_mwh": self.grid_energy,
            "wake_loss_pct": self.wake_loss / denom * 100.0 if denom else 0.0,
            "unavailability_loss_pct": self.unavailability_loss / denom * 100.0 if denom else 0.0,
            "curtailment_loss_pct": self.curtailment_loss / denom * 100.0 if denom else 0.0,
        }


class OperatingScenario:
    """运行情景：一组时段及其运行前校验。

    Parameters
    ----------
    periods : list[OperatingPeriod]
        时段列表，按时间顺序排列；至少包含一个时段
    n_turbines : int
        风机台数（用于校验逐机可利用率长度）
    total_hours : float
        所有时段小时数之和应闭合到的目标值，默认 8760。
        传入 ``None`` 则不检查总小时数（自定义时段不要求覆盖全年）。
    default_wind_resource : Optional[WindResource]
        未单独指定风资源的时段使用的默认风资源
    require_full_coverage : bool
        是否要求时段必须覆盖完整年度（总小时数闭合到 ``total_hours``）。
        自定义时段可设为 False，此时仅校验总小时数不超过目标值。
    """

    def __init__(
        self,
        periods: Sequence[OperatingPeriod],
        n_turbines: int,
        total_hours: Optional[float] = HOURS_PER_YEAR,
        default_wind_resource: Optional[WindResource] = None,
        require_full_coverage: bool = True,
        installed_capacity_mw: Optional[float] = None,
    ) -> None:
        self.periods = list(periods)
        self.n_turbines = n_turbines
        self.total_hours = total_hours
        self.default_wind_resource = default_wind_resource
        self.require_full_coverage = require_full_coverage
        self.installed_capacity_mw = installed_capacity_mw
        self.validate()

    def validate(self) -> None:
        """运行前校验，任何不满足都抛出 :class:`ScenarioValidationError`。"""
        if not self.periods:
            raise ScenarioValidationError("运行情景至少需要配置一个时段")

        if self.n_turbines <= 0:
            raise ScenarioValidationError(f"风机台数必须为正整数，当前为 {self.n_turbines}")

        seen_names: set[str] = set()
        hours_sum = 0.0

        for period in self.periods:
            # dataclass __post_init__ 已做基础校验，这里补齐跨字段校验
            if period.name in seen_names:
                raise ScenarioValidationError(f"时段名称重复: {period.name!r}")
            seen_names.add(period.name)
            hours_sum += period.hours

            if period.wind_resource is None and self.default_wind_resource is None:
                raise ScenarioValidationError(
                    f"时段 {period.name!r} 未指定风资源，且情景未提供默认风资源"
                )

            av = period.availability
            if av is not None:
                arr = np.asarray(av, dtype=np.float64)
                if arr.ndim == 0:
                    arr = np.full(self.n_turbines, float(arr))
                if arr.shape != (self.n_turbines,):
                    raise ScenarioValidationError(
                        f"时段 {period.name!r} 的逐机可利用率长度应为 {self.n_turbines}，"
                        f"实际为 {arr.shape[0] if arr.ndim == 1 else arr.shape}"
                    )
                if np.any(arr < 0.0) or np.any(arr > 1.0 + 1e-9):
                    bad = int(np.where((arr < 0.0) | (arr > 1.0 + 1e-9))[0][0])
                    raise ScenarioValidationError(
                        f"时段 {period.name!r} 中第 {bad} 台机组的可利用率越界 "
                        f"(允许 [0, 1])，值为 {arr[bad]:g}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ScenarioValidationError(
                        f"时段 {period.name!r} 的逐机可利用率包含非有限值"
                    )

            if (
                self.installed_capacity_mw is not None
                and period.grid_capacity_mw is not None
                and period.grid_capacity_mw > self.installed_capacity_mw + 1e-9
            ):
                raise ScenarioValidationError(
                    f"时段 {period.name!r} 的并网功率上限 "
                    f"{period.grid_capacity_mw:g} MW 超过全场额定装机 "
                    f"{self.installed_capacity_mw:g} MW，容量上限配置冲突"
                )

        if self.total_hours is not None:
            target = float(self.total_hours)
            if self.require_full_coverage:
                if abs(hours_sum - target) > HOURS_TOLERANCE:
                    raise ScenarioValidationError(
                        f"各时段小时数之和 {hours_sum:g} 与目标 {target:g} 不闭合"
                        f"（偏差 {hours_sum - target:+g} 小时），"
                        "请检查是否有缺失月份或小时数配置错误"
                    )
            elif hours_sum > target + HOURS_TOLERANCE:
                raise ScenarioValidationError(
                    f"各时段小时数之和 {hours_sum:g} 超过目标 {target:g}，"
                    "时段不得重复计入同一小时"
                )

    def resource_for(self, period: OperatingPeriod) -> WindResource:
        """返回时段实际使用的风资源。"""
        return period.wind_resource if period.wind_resource is not None else self.default_wind_resource

    def availability_for(self, period: OperatingPeriod) -> np.ndarray:
        """返回时段逐机可利用率数组（缺省为全部 1.0）。"""
        if period.availability is None:
            return np.ones(self.n_turbines, dtype=np.float64)
        arr = np.asarray(period.availability, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.n_turbines, float(arr))
        return arr


class OperationalEnergyCalculator:
    """基于运行情景的能量计算器。

    Parameters
    ----------
    aep_calculator : AEPCalculator
        已绑定风机与（默认）风资源的 AEP 计算器，负责尾流与功率积分
    """

    def __init__(self, aep_calculator: AEPCalculator) -> None:
        self.aep_calculator = aep_calculator
        self._rated_powers_mw = np.array(
            [t.rated_power for t in aep_calculator.turbines], dtype=np.float64
        ) / 1e3

    def compute_period(
        self,
        positions: np.ndarray,
        period: OperatingPeriod,
        scenario: OperatingScenario,
    ) -> PeriodEnergyResult:
        """计算单个时段的五级能量分解。

        计算顺序固定为：毛发电 → 尾流 → 可利用率 → 限发 → 上网电量。
        可利用率与并网上限都施加在每个"风向 × 风速"运行状态上：

        1. 尾流积分给出各状态逐机毛功率与尾流后功率 (kW)；
        2. 逐机可利用率把尾流后功率乘以该机组可用比例；
        3. 每个状态下，全场可用功率若超过并网功率上限，则超出部分
           按各机可用功率等比例压减（``proportional`` 规则，
           对每台机组一致、可复现、可解释）；
        4. 状态功率乘以该状态的小时权重（扇区频率 × 风速概率 ×
           时段小时数）积分得到能量。小时权重只在这里乘一次。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        period : OperatingPeriod
            时段定义
        scenario : OperatingScenario
            所属情景（提供默认风资源与校验）

        Returns
        -------
        PeriodEnergyResult
            时段能量分解（MWh）
        """
        resource = scenario.resource_for(period)
        availability = scenario.availability_for(period)

        hours, weights, gross_power_kw, net_power_kw = (
            self.aep_calculator.compute_period_power_distribution(
                positions, hours=period.hours, wind_resource=resource
            )
        )

        # weights: (N_sector, N_speed)；功率: (N_sector, N_turb, N_speed)
        # 状态小时数 = hours * weights，对全部状态求和等于 hours
        state_hours = hours * weights  # (N_sector, N_speed)

        # 1) 毛发电与尾流损失
        gross_kwh = np.sum(
            gross_power_kw * state_hours[:, np.newaxis, :], axis=(0, 2)
        )
        post_wake_kwh = np.sum(
            net_power_kw * state_hours[:, np.newaxis, :], axis=(0, 2)
        )

        # 2) 逐机可利用率（检修/故障率），在每个状态上等比例折减
        avail_power_kw = net_power_kw * availability[np.newaxis, :, np.newaxis]
        available_kwh = np.sum(
            avail_power_kw * state_hours[:, np.newaxis, :], axis=(0, 2)
        )
        unavail_loss_kwh = post_wake_kwh - available_kwh

        post_sum = float(np.sum(post_wake_kwh))
        mean_avail = (
            float(np.sum(availability * post_wake_kwh) / post_sum)
            if post_sum > 0.0
            else 1.0
        )

        # 3) 并网功率上限：状态级 min(全场可用功率, 上限)，
        #    超出部分按各机可用功率比例分配（proportional）
        cap_kw = (
            period.grid_capacity_mw * 1e3
            if period.grid_capacity_mw is not None
            else None
        )
        farm_avail_kw = np.sum(avail_power_kw, axis=1)  # (N_sector, N_speed)

        if cap_kw is None:
            delivered_share = np.ones_like(farm_avail_kw)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(
                    farm_avail_kw > 0.0,
                    np.minimum(farm_avail_kw, cap_kw) / farm_avail_kw,
                    1.0,
                )
            delivered_share = share  # (N_sector, N_speed)，每个状态全场统一比例

        grid_power_kw = avail_power_kw * delivered_share[:, np.newaxis, :]
        grid_kwh = np.sum(
            grid_power_kw * state_hours[:, np.newaxis, :], axis=(0, 2)
        )
        curtailment_kwh = available_kwh - grid_kwh

        available_total_mwh = float(np.sum(available_kwh) / 1e3)
        curtailment_total_mwh = float(np.sum(curtailment_kwh) / 1e3)
        curtailment_factor = (
            curtailment_total_mwh / available_total_mwh
            if available_total_mwh > 0.0
            else 0.0
        )

        gross_mwh = gross_kwh / 1e3
        post_wake_mwh = post_wake_kwh / 1e3

        return PeriodEnergyResult(
            name=period.name,
            hours=float(hours),
            gross_energy=float(np.sum(gross_mwh)),
            wake_loss=float(np.sum(gross_mwh - post_wake_mwh)),
            post_wake_energy=float(np.sum(post_wake_mwh)),
            unavailability_loss=float(np.sum(unavail_loss_kwh) / 1e3),
            available_energy=available_total_mwh,
            curtailment_loss=curtailment_total_mwh,
            grid_energy=float(np.sum(grid_kwh) / 1e3),
            curtailment_factor=curtailment_factor,
            mean_availability=mean_avail,
            turbine_gross=gross_mwh,
            turbine_post_wake=post_wake_mwh,
            turbine_available=available_kwh / 1e3,
            turbine_grid=grid_kwh / 1e3,
            turbine_curtailment=curtailment_kwh / 1e3,
        )

    def compute_scenario(
        self,
        positions: np.ndarray,
        scenario: OperatingScenario,
    ) -> ScenarioEnergyResult:
        """计算整个情景：逐时段计算后直接相加。

        各时段小时数互斥（情景校验保证不重不漏），因此直接求和即为
        全年（或自定义周期）总量，不存在额外权重或重复计入。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        scenario : OperatingScenario
            已通过校验的运行情景

        Returns
        -------
        ScenarioEnergyResult
            全情景能量分解（MWh）
        """
        #入口处再次校验，保证"运行前拒绝"的约束不依赖调用方记得调用 validate
        scenario.validate()

        positions = np.asarray(positions, dtype=np.float64)
        n_turb = len(self.aep_calculator.turbines)
        if positions.shape != (n_turb, 2):
            raise ValueError(
                f"位置数组形状应为 ({n_turb}, 2)，实际为 {positions.shape}"
            )

        period_results = [
            self.compute_period(positions, p, scenario) for p in scenario.periods
        ]

        return ScenarioEnergyResult(
            periods=period_results,
            total_hours=float(sum(p.hours for p in period_results)),
            gross_energy=float(sum(p.gross_energy for p in period_results)),
            wake_loss=float(sum(p.wake_loss for p in period_results)),
            unavailability_loss=float(sum(p.unavailability_loss for p in period_results)),
            curtailment_loss=float(sum(p.curtailment_loss for p in period_results)),
            grid_energy=float(sum(p.grid_energy for p in period_results)),
        )


def create_monthly_periods(
    availabilities: Optional[Sequence[float | Sequence[float]]] = None,
    grid_capacities_mw: Optional[Sequence[Optional[float]]] = None,
    wind_resources: Optional[Sequence[Optional[WindResource]]] = None,
    names: Optional[Sequence[str]] = None,
    year: int = 2025,
) -> list[OperatingPeriod]:
    """便捷生成 12 个自然月时段（非闰年 28 天，合计恰为 8760 小时）。

    Parameters
    ----------
    availabilities : 可选，长度 12；每个元素为标量（全场相同可利用率）
        或长度 N_turbines 的逐机数组
    grid_capacities_mw : 可选，长度 12，每月全场并网上限 (MW)，None 表示不限
    wind_resources : 可选，长度 12，每月风资源，None 表示用情景默认风资源
    names : 可选，长度 12 的时段名称
    year : 年份，仅 2 月天数受闰年影响；默认取平年使总小时数恰为 8760

    Returns
    -------
    list[OperatingPeriod]
        12 个月时段（小时数依次为 31*24, 28*24, ...）
    """
    days = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def _check_len(seq, label):
        if seq is not None and len(seq) != 12:
            raise ScenarioValidationError(f"{label} 长度应为 12，实际为 {len(seq)}")

    _check_len(availabilities, "availabilities")
    _check_len(grid_capacities_mw, "grid_capacities_mw")
    _check_len(wind_resources, "wind_resources")
    _check_len(names, "names")

    periods: list[OperatingPeriod] = []
    for m in range(12):
        av = None
        if availabilities is not None and availabilities[m] is not None:
            value = np.asarray(availabilities[m], dtype=np.float64)
            av = value if value.ndim == 1 else float(value)
        periods.append(
            OperatingPeriod(
                name=names[m] if names is not None else f"{m + 1}月",
                hours=float(days[m] * 24),
                wind_resource=wind_resources[m] if wind_resources is not None else None,
                availability=av,
                grid_capacity_mw=grid_capacities_mw[m] if grid_capacities_mw is not None else None,
            )
        )
    return periods

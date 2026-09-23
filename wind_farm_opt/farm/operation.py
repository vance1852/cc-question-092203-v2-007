"""分时段运行情景与电量平衡计算。

在风资源与尾流/AEP 流程之上引入按月（或任意自定义时段）的运行情景：
每个时段独立指定小时数、风资源、逐机可利用率与全场并网功率上限。

电量按以下顺序形成，五层口径互不重叠：

    毛发电 gross
      -> 扣尾流损失 wake_loss
    尾流后电量（可用机组满发）
      -> 扣不可利用损失 availability_loss（机组检修/故障停机）
    可用电量 available
      -> 扣限发损失 curtailment_loss（全场并网功率上限）
    最终上网电量 delivered

设计约定：
- 每个时段的小时数在 ``AEPCalculator.compute_period_energy`` 中恰好乘一次；
  时段权重只用于"代表性时段"的年电量放大，二者不会同时作用于同一电量。
- 限发按可解释的规则在机组间分配：默认按各机组可用发电功率等比例削减
  （proportional），也支持按机组顺序优先保电（merit_order）。
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.wind_resource import WindResource, WindSector
from .aep import AEPCalculator

#: 全年小时数
ANNUAL_HOURS = 8760.0

#: 非闰年各月小时数（合计恰为 8760）
MONTH_HOURS_NON_LEAP = (
    744.0, 672.0, 744.0, 720.0, 744.0, 720.0,
    744.0, 744.0, 720.0, 744.0, 720.0, 744.0,
)

#: 支持的限发分配策略
CURTAILMENT_STRATEGIES = ("proportional", "merit_order")


def scale_wind_resource(base: WindResource, speed_factor: float) -> WindResource:
    """按比例缩放风资源各扇区的风速。

    保持风向频率与威布尔形状参数 k 不变，将尺度参数 c 与平均风速乘以
    ``speed_factor``，用于从基准风玫瑰派生逐月（或逐时段）风资源。

    Parameters
    ----------
    base : WindResource
        基准风资源
    speed_factor : float
        风速缩放比例（必须为正），1.0 表示与基准一致

    Returns
    -------
    WindResource
        缩放后的新风资源
    """
    if not np.isfinite(speed_factor) or speed_factor <= 0:
        raise ValueError(f"风速缩放比例必须为正数，当前为 {speed_factor}")

    sectors = [
        WindSector(
            direction_center=s.direction_center,
            direction_width=s.direction_width,
            frequency=s.frequency,
            mean_speed=s.mean_speed * speed_factor,
            weibull_k=s.weibull_k,
            weibull_c=s.weibull_c * speed_factor,
        )
        for s in base.sectors
    ]
    return WindResource(sectors)


@dataclass
class OperatingPeriod:
    """单个运行时段（如一个月或任意自定义时段）。

    Parameters
    ----------
    name : str
        时段名称（如 "1月"、"冬季检修期"）
    hours : float
        时段实际小时数（非闰年月度可由日历给出）。
    wind_resource : WindResource
        该时段的风资源（扇区频率在时段内部归一化，和为 1）。
    availability : np.ndarray
        逐机可利用率，形状 (N_turb,)，取值 [0, 1]；
        1 表示全时段可用，0 表示全时段停机检修。
    grid_capacity_mw : Optional[float]
        全场并网功率上限 (MW)；None 表示不限制送出。
    weight : float
        时段年重复次数。普通逐月情景为 1.0（各时段小时数之和应为 8760）；
        自定义"代表性时段"时可取 >0 的非整数，使 weight*hours 构成全年覆盖，
        此时 hours 本身不再额外计入（避免小时数与权重重复计入）。
    curtailment_strategy : str
        限发分配策略："proportional"（按可用功率等比例）或
        "merit_order"（按给定机组顺序优先保电）。
    merit_order : Optional[Sequence[int]]
        merit_order 策略下的机组优先级（索引从高到低）；缺省按机组索引。
    """

    name: str
    hours: float
    wind_resource: WindResource
    availability: np.ndarray
    grid_capacity_mw: Optional[float] = None
    weight: float = 1.0
    curtailment_strategy: str = "proportional"
    merit_order: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        self.availability = np.asarray(self.availability, dtype=np.float64)

    def validate(self, n_turbines: int) -> None:
        """逐时段静态校验（不依赖其他时段）。

        Parameters
        ----------
        n_turbines : int
            机组台数

        Raises
        ------
        ValueError
            小时数、权重、可利用率或并网上限不合法时抛出。
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("时段名称不能为空")

        if not np.isfinite(self.hours) or self.hours <= 0:
            raise ValueError(
                f"时段 {self.name!r} 的小时数必须为正数，当前为 {self.hours}"
            )

        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(
                f"时段 {self.name!r} 的权重必须为正数，当前为 {self.weight}"
            )

        if self.availability.shape != (n_turbines,):
            raise ValueError(
                f"时段 {self.name!r} 的可利用率数组形状应为 ({n_turbines},)，"
                f"实际为 {self.availability.shape}"
            )
        if not np.all(np.isfinite(self.availability)):
            raise ValueError(f"时段 {self.name!r} 的可利用率包含非数值")
        if np.any(self.availability < 0.0) or np.any(self.availability > 1.0):
            bad = self.availability[
                (self.availability < 0.0) | (self.availability > 1.0)
            ]
            raise ValueError(
                f"时段 {self.name!r} 存在负的或超过1的可利用率: {bad.tolist()}"
            )

        if self.grid_capacity_mw is not None:
            cap = float(self.grid_capacity_mw)
            if not np.isfinite(cap) or cap < 0:
                raise ValueError(
                    f"时段 {self.name!r} 的并网功率上限必须为非负数值或 null，"
                    f"当前为 {self.grid_capacity_mw}"
                )

        if self.curtailment_strategy not in CURTAILMENT_STRATEGIES:
            raise ValueError(
                f"时段 {self.name!r} 的限发策略 {self.curtailment_strategy!r} 不支持，"
                f"可选: {CURTAILMENT_STRATEGIES}"
            )

        if self.curtailment_strategy == "merit_order":
            order = list(range(n_turbines)) if self.merit_order is None else list(self.merit_order)
            if sorted(order) != list(range(n_turbines)):
                raise ValueError(
                    f"时段 {self.name!r} 的 merit_order 必须是 0..{n_turbines - 1} "
                    f"的一个排列，当前为 {self.merit_order}"
                )


@dataclass
class PeriodEnergyResult:
    """单个时段的电量平衡结果（单位均为 MWh）。

    口径关系：
        gross - wake_loss = 尾流后电量
        尾流后电量 - availability_loss = available
        available - curtailment_loss = delivered
        gross - wake_loss - availability_loss - curtailment_loss = delivered
    """

    name: str
    hours: float
    weight: float
    gross: float
    wake_loss: float
    availability_loss: float
    curtailment_loss: float
    delivered: float
    grid_capacity_mw: Optional[float]
    curtailment_strategy: str
    delivered_by_turbine: np.ndarray
    curtailment_by_turbine: np.ndarray

    @property
    def available(self) -> float:
        """扣完尾流与不可利用损失后的可用电量 (MWh)。"""
        return self.gross - self.wake_loss - self.availability_loss

    def as_dict(self) -> dict:
        """转为可序列化字典（MWh）。"""
        return {
            "name": self.name,
            "hours": float(self.hours),
            "weight": float(self.weight),
            "grid_capacity_mw": (
                float(self.grid_capacity_mw)
                if self.grid_capacity_mw is not None
                else None
            ),
            "curtailment_strategy": self.curtailment_strategy,
            "gross_mwh": float(self.gross),
            "wake_loss_mwh": float(self.wake_loss),
            "availability_loss_mwh": float(self.availability_loss),
            "available_mwh": float(self.available),
            "curtailment_loss_mwh": float(self.curtailment_loss),
            "delivered_mwh": float(self.delivered),
            "delivered_by_turbine_mwh": np.asarray(
                self.delivered_by_turbine, dtype=np.float64
            ).tolist(),
            "curtailment_by_turbine_mwh": np.asarray(
                self.curtailment_by_turbine, dtype=np.float64
            ).tolist(),
        }


@dataclass
class ScenarioEnergyResult:
    """全年（所有权重放大后）分时段情景的电量平衡汇总（单位均为 MWh）。"""

    period_results: list[PeriodEnergyResult]
    total_installed_capacity_mw: float
    gross_mwh: float
    wake_loss_mwh: float
    availability_loss_mwh: float
    curtailment_loss_mwh: float
    delivered_mwh: float
    total_hours: float

    @property
    def available_mwh(self) -> float:
        """全年可用电量（扣尾流、扣不可利用，未扣限发）(MWh)。"""
        return (
            self.gross_mwh
            - self.wake_loss_mwh
            - self.availability_loss_mwh
        )

    @property
    def net_energy_mwh(self) -> float:
        """最终上网电量（交给经济分析的口径）(MWh)。"""
        return self.delivered_mwh

    def as_dict(self) -> dict:
        """转为可序列化字典。"""
        def pct(part: float) -> Optional[float]:
            return float(part / self.gross_mwh * 100.0) if self.gross_mwh > 0 else 0.0

        return {
            "total_hours": float(self.total_hours),
            "total_installed_capacity_mw": float(self.total_installed_capacity_mw),
            "gross_mwh": float(self.gross_mwh),
            "wake_loss_mwh": float(self.wake_loss_mwh),
            "availability_loss_mwh": float(self.availability_loss_mwh),
            "available_mwh": float(self.available_mwh),
            "curtailment_loss_mwh": float(self.curtailment_loss_mwh),
            "delivered_mwh": float(self.delivered_mwh),
            "wake_loss_pct": pct(self.wake_loss_mwh),
            "availability_loss_pct": pct(self.availability_loss_mwh),
            "curtailment_loss_pct": pct(self.curtailment_loss_mwh),
            "delivered_pct": pct(self.delivered_mwh),
            "periods": [p.as_dict() for p in self.period_results],
        }


class OperatingScenarioSet:
    """运行情景集合：一组分时段定义及其全年覆盖校验。

    Parameters
    ----------
    periods : list[OperatingPeriod]
        时段列表
    annual_hours : float
        全年应覆盖小时数，默认 8760
    hours_tolerance : float
        逐月情景（权重均为 1）下小时数闭合容差
    """

    def __init__(
        self,
        periods: list[OperatingPeriod],
        annual_hours: float = ANNUAL_HOURS,
        hours_tolerance: float = 1.0,
    ) -> None:
        self.periods = list(periods)
        self.annual_hours = float(annual_hours)
        self.hours_tolerance = float(hours_tolerance)

    @property
    def weighted_hours(self) -> float:
        """权重放大后的总覆盖小时数。"""
        return float(sum(p.hours * p.weight for p in self.periods))

    def validate(
        self,
        n_turbines: int,
        installed_capacity_mw: Optional[float] = None,
    ) -> None:
        """运行前的全部配置校验。

        逐项检查：空时段、缺失（逐时段静态字段）、权重闭合、负可利用率、
        容量上限冲突等，任何一项不合法都在运行前抛出 ``ValueError``。

        Parameters
        ----------
        n_turbines : int
            机组台数
        installed_capacity_mw : Optional[float]
            全场装机容量 (MW)，用于检查并网功率上限是否与装机容量冲突

        Raises
        ------
        ValueError
            配置不合法时抛出，错误信息指明具体时段与原因。
        """
        if not self.periods:
            raise ValueError("运行情景至少需要配置一个时段（当前为空）")

        names = [p.name for p in self.periods]
        if len(set(names)) != len(names):
            dup = {n for n in names if names.count(n) > 1}
            raise ValueError(f"运行情景中存在重名时段: {sorted(dup)}")

        for period in self.periods:
            period.validate(n_turbines)

            if (
                installed_capacity_mw is not None
                and period.grid_capacity_mw is not None
                and period.grid_capacity_mw > installed_capacity_mw + 1e-9
            ):
                raise ValueError(
                    f"时段 {period.name!r} 的并网功率上限 "
                    f"{period.grid_capacity_mw:.3f} MW 超过全场装机容量 "
                    f"{installed_capacity_mw:.3f} MW，该上限永远不会触发限发，"
                    f"属于容量上限冲突（请检查单位或数值）"
                )

        total = self.weighted_hours
        if abs(total - self.annual_hours) > self.hours_tolerance:
            raise ValueError(
                f"运行情景权重不闭合：各时段 hours*weight 之和为 {total:.2f} 小时，"
                f"应为全年 {self.annual_hours:.0f} 小时（容差 ±{self.hours_tolerance:g}）。"
                f"逐月情景请检查缺失/多余月份；代表性时段请调整 weight。"
            )


def allocate_curtailment(
    available_power_mw: np.ndarray,
    hours: float,
    grid_capacity_mw: Optional[float],
    strategy: str = "proportional",
    merit_order: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """按可解释规则把全场送出限制分配到各机组。

    以时段平均功率刻画限发：若各机组可用功率之和超过并网上限，则按策略
    削减各机组上网电量，削减量恰好等于超限部分（能量守恒）。

    Parameters
    ----------
    available_power_mw : np.ndarray
        各机组时段平均可用功率 (MW)，已含可利用率折减
    hours : float
        时段小时数
    grid_capacity_mw : Optional[float]
        全场并网功率上限 (MW)；None 或足够大时不限发
    strategy : str
        "proportional"：所有超限机组按可用功率同比例削减；
        "merit_order"：优先保证排序靠前机组满发，超限电量由排序靠后机组承担
    merit_order : Optional[Sequence[int]]
        merit_order 下的机组优先级（靠前优先保电）

    Returns
    -------
    np.ndarray
        各机组被限发的电量 (MWh)，非负且总和等于全场限发量
    """
    n = available_power_mw.shape[0]
    total_power = float(np.sum(available_power_mw))

    if grid_capacity_mw is None or total_power <= float(grid_capacity_mw):
        return np.zeros(n, dtype=np.float64)

    excess_mw = total_power - float(grid_capacity_mw)

    if strategy == "proportional":
        share = available_power_mw / total_power
        return share * excess_mw * hours

    # merit_order：从优先级最低的机组开始承担限发，依次向上
    order = (
        list(range(n))
        if merit_order is None
        else list(merit_order)
    )
    curtail_mw = np.zeros(n, dtype=np.float64)
    remaining = excess_mw
    for idx in reversed(order):
        take = min(remaining, float(available_power_mw[idx]))
        curtail_mw[idx] = take
        remaining -= take
        if remaining <= 1e-9:
            break

    return curtail_mw * hours


class ScenarioEnergyCalculator:
    """分时段情景电量计算器。

    包装 :class:`AEPCalculator`，对每个时段依次执行：
    尾流后机组功率/电量 -> 逐机可利用率 -> 全场限发分配。

    Parameters
    ----------
    aep_calculator : AEPCalculator
        已绑定机组、（基准）风资源与尾流模型的 AEP 计算器
    """

    def __init__(self, aep_calculator: AEPCalculator) -> None:
        self.aep_calculator = aep_calculator
        self.n_turbines = len(aep_calculator.turbines)
        self.rated_powers_mw = aep_calculator._rated_powers / 1e3

    def compute_period(
        self,
        positions: np.ndarray,
        period: OperatingPeriod,
    ) -> PeriodEnergyResult:
        """计算单个时段的电量平衡（未做年权重放大）。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        period : OperatingPeriod
            运行时段定义

        Returns
        -------
        PeriodEnergyResult
            时段电量平衡（单位 MWh，对应一个 ``hours`` 时段）
        """
        # 1) 先形成尾流后的机组电量（小时数在此处只乘这一次）
        gross_mwh, post_wake_mwh = self.aep_calculator.compute_period_energy(
            positions,
            period.wind_resource,
            period.hours,
        )

        wake_loss_mwh = gross_mwh - post_wake_mwh

        # 2) 再应用逐机可利用率（检修/故障）
        delivered_pre_curtail_mwh = post_wake_mwh * period.availability
        availability_loss_mwh = post_wake_mwh - delivered_pre_curtail_mwh

        # 3) 按可解释规则进行全场送出限发分配
        avg_power_mw = (
            delivered_pre_curtail_mwh / period.hours
            if period.hours > 0
            else np.zeros_like(delivered_pre_curtail_mwh)
        )
        curtailment_mwh = allocate_curtailment(
            avg_power_mw,
            hours=period.hours,
            grid_capacity_mw=period.grid_capacity_mw,
            strategy=period.curtailment_strategy,
            merit_order=period.merit_order,
        )
        # 数值保护：限发不得超过该机组可用电量
        curtailment_mwh = np.minimum(curtailment_mwh, delivered_pre_curtail_mwh)

        delivered_mwh_by_turbine = delivered_pre_curtail_mwh - curtailment_mwh

        return PeriodEnergyResult(
            name=period.name,
            hours=float(period.hours),
            weight=float(period.weight),
            gross=float(np.sum(gross_mwh)),
            wake_loss=float(np.sum(wake_loss_mwh)),
            availability_loss=float(np.sum(availability_loss_mwh)),
            curtailment_loss=float(np.sum(curtailment_mwh)),
            delivered=float(np.sum(delivered_mwh_by_turbine)),
            grid_capacity_mw=(
                float(period.grid_capacity_mw)
                if period.grid_capacity_mw is not None
                else None
            ),
            curtailment_strategy=period.curtailment_strategy,
            delivered_by_turbine=delivered_mwh_by_turbine,
            curtailment_by_turbine=curtailment_mwh,
        )

    def compute(
        self,
        positions: np.ndarray,
        scenario: OperatingScenarioSet,
    ) -> ScenarioEnergyResult:
        """计算全情景（运行前校验 -> 逐时段 -> 年权重汇总）。

        权重只在汇总时施加于已积分好的时段电量，因此不会与时段小时数
        重复计入。逐月情景权重均为 1，等价于直接相加。

        Parameters
        ----------
        positions : np.ndarray
            风机位置 (N_turb, 2)
        scenario : OperatingScenarioSet
            运行情景集合

        Returns
        -------
        ScenarioEnergyResult
            全年电量平衡汇总
        """
        # 所有配置问题必须在任何电量计算之前暴露
        scenario.validate(
            self.n_turbines,
            installed_capacity_mw=float(np.sum(self.rated_powers_mw)),
        )

        period_results = [
            self.compute_period(positions, period) for period in scenario.periods
        ]

        gross = sum(p.gross * p.weight for p in period_results)
        wake_loss = sum(p.wake_loss * p.weight for p in period_results)
        availability_loss = sum(
            p.availability_loss * p.weight for p in period_results
        )
        curtailment_loss = sum(
            p.curtailment_loss * p.weight for p in period_results
        )
        delivered = sum(p.delivered * p.weight for p in period_results)

        return ScenarioEnergyResult(
            period_results=period_results,
            total_installed_capacity_mw=float(np.sum(self.rated_powers_mw)),
            gross_mwh=float(gross),
            wake_loss_mwh=float(wake_loss),
            availability_loss_mwh=float(availability_loss),
            curtailment_loss_mwh=float(curtailment_loss),
            delivered_mwh=float(delivered),
            total_hours=scenario.weighted_hours,
        )


_DEFAULT_MONTH_NAMES = (
    "1月", "2月", "3月", "4月", "5月", "6月",
    "7月", "8月", "9月", "10月", "11月", "12月",
)


def _broadcast_availability(
    value, n_turbines: int
) -> np.ndarray:
    """把标量或长度 N 的序列规范为 (N_turb,) 可利用率数组。"""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_turbines, float(arr))
    if arr.ndim == 1 and arr.shape[0] == n_turbines:
        return arr
    raise ValueError(
        f"可利用率应为标量或长度为 {n_turbines} 的数组，实际形状为 {arr.shape}"
    )


def build_monthly_scenario(
    base_wind_resource: WindResource,
    n_turbines: int,
    availability: float | Sequence | np.ndarray = 1.0,
    grid_capacity_mw: Optional[float | Sequence[Optional[float]]] = None,
    speed_factors: Optional[Sequence[float]] = None,
    curtailment_strategy: str = "proportional",
    merit_order: Optional[Sequence[int]] = None,
    month_names: Sequence[str] = _DEFAULT_MONTH_NAMES,
) -> OperatingScenarioSet:
    """便捷构建 12 个逐月运行时段。

    各月使用非闰年自然月小时数（合计恰为 8760）。

    Parameters
    ----------
    base_wind_resource : WindResource
        基准风资源；逐月风资源由 ``speed_factors`` 在此基础上缩放
    n_turbines : int
        机组台数
    availability : float | Sequence | np.ndarray
        逐机可利用率，支持：
        标量（全场全年相同）、长度12的逐月标量、形状 (12, N_turb) 的逐月逐机数组
    grid_capacity_mw : Optional[float | Sequence]
        全场并网功率上限 (MW)，可为 None（不限）、标量或长度12的逐月数值
    speed_factors : Optional[Sequence[float]]
        长度12的逐月风速缩放比例，缺省各月均为 1.0
    curtailment_strategy : str
        限发分配策略
    merit_order : Optional[Sequence[int]]
        merit_order 策略下的机组优先级
    month_names : Sequence[str]
        12 个时段名称

    Returns
    -------
    OperatingScenarioSet
        月度运行情景（尚未运行，可在计算前统一校验）
    """
    avail = np.asarray(availability, dtype=np.float64)
    if avail.ndim == 0:
        avail_grid = np.full((12, n_turbines), float(avail))
    elif avail.ndim == 1 and avail.shape[0] == 12:
        avail_grid = np.repeat(avail[:, np.newaxis], n_turbines, axis=1)
    elif avail.ndim == 2 and avail.shape == (12, n_turbines):
        avail_grid = avail
    else:
        raise ValueError(
            "availability 应为标量、长度12的逐月标量或形状 (12, N_turb) 的数组，"
            f"实际形状为 {avail.shape}"
        )

    if grid_capacity_mw is None or np.isscalar(grid_capacity_mw):
        caps = [grid_capacity_mw] * 12
    else:
        caps = list(grid_capacity_mw)
        if len(caps) != 12:
            raise ValueError(f"grid_capacity_mw 序列长度应为12，实际为 {len(caps)}")

    if speed_factors is None:
        factors = [1.0] * 12
    else:
        factors = list(speed_factors)
        if len(factors) != 12:
            raise ValueError(f"speed_factors 长度应为12，实际为 {len(factors)}")

    if len(month_names) != 12:
        raise ValueError(f"month_names 长度应为12，实际为 {len(month_names)}")

    periods = []
    for m in range(12):
        wr = (
            base_wind_resource
            if factors[m] == 1.0
            else scale_wind_resource(base_wind_resource, factors[m])
        )
        periods.append(
            OperatingPeriod(
                name=str(month_names[m]),
                hours=MONTH_HOURS_NON_LEAP[m],
                wind_resource=wr,
                availability=avail_grid[m],
                grid_capacity_mw=(None if caps[m] is None else float(caps[m])),
                weight=1.0,
                curtailment_strategy=curtailment_strategy,
                merit_order=merit_order,
            )
        )

    return OperatingScenarioSet(periods)


def _period_wind_resource(
    spec, base_wind_resource: WindResource
) -> WindResource:
    """根据时段配置解析风资源。

    支持：None/"base"（直接使用基准风资源）、{"speed_factor": x} 缩放、
    {"type": "default"/"uniform", ...} 新建风资源。
    """
    if spec is None:
        return base_wind_resource
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"时段风资源配置应为字典，当前为 {type(spec)!r}")

    kind = spec.get("type", "speed_factor" if "speed_factor" in spec else "base")

    if kind == "base":
        return base_wind_resource
    if kind == "speed_factor":
        return scale_wind_resource(base_wind_resource, float(spec["speed_factor"]))
    if kind == "default":
        from ..core.wind_resource import create_default_wind_resource
        return create_default_wind_resource(
            num_sectors=spec.get("num_sectors", 12),
            dominant_direction=spec.get("dominant_direction", 270.0),
            mean_speed=spec.get("mean_speed", 8.5),
        )
    if kind == "uniform":
        from ..core.wind_resource import create_simple_wind_resource
        return create_simple_wind_resource(
            num_sectors=spec.get("num_sectors", 12),
            uniform=True,
            mean_speed=spec.get("mean_speed", 8.0),
        )
    raise ValueError(f"未知的时段风资源类型: {kind!r}")


def build_scenario_from_dicts(
    period_dicts: Sequence[dict],
    n_turbines: int,
    base_wind_resource: WindResource,
    annual_hours: float = ANNUAL_HOURS,
) -> OperatingScenarioSet:
    """从 JSON 兼容的字典列表构建自定义运行情景。

    每个时段字典支持的键：name、hours（必填）、wind_resource、availability、
    grid_capacity_mw、weight、curtailment_strategy、merit_order。

    Parameters
    ----------
    period_dicts : Sequence[dict]
        自定义时段定义
    n_turbines : int
        机组台数
    base_wind_resource : WindResource
        基准风资源
    annual_hours : float
        全年应覆盖小时数（用于权重闭合校验）

    Returns
    -------
    OperatingScenarioSet
        自定义运行情景
    """
    if not period_dicts:
        raise ValueError("自定义运行情景的时段列表为空")

    periods = []
    for i, d in enumerate(period_dicts):
        if not isinstance(d, dict):
            raise ValueError(f"第 {i + 1} 个时段配置必须为字典")
        if "name" not in d:
            raise ValueError(f"第 {i + 1} 个时段缺少 name")
        if "hours" not in d:
            raise ValueError(f"时段 {d.get('name')!r} 缺少 hours")

        availability = d.get("availability", 1.0)
        avail_arr = _broadcast_availability(availability, n_turbines)

        periods.append(
            OperatingPeriod(
                name=str(d["name"]),
                hours=float(d["hours"]),
                wind_resource=_period_wind_resource(
                    d.get("wind_resource"), base_wind_resource
                ),
                availability=avail_arr,
                grid_capacity_mw=(
                    None
                    if d.get("grid_capacity_mw") is None
                    else float(d["grid_capacity_mw"])
                ),
                weight=float(d.get("weight", 1.0)),
                curtailment_strategy=d.get("curtailment_strategy", "proportional"),
                merit_order=d.get("merit_order"),
            )
        )

    return OperatingScenarioSet(periods, annual_hours=annual_hours)

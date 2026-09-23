# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 分时段运行情景（检修、逐机可利用率、送出限发）

默认流程把全年 8760 小时都视为机组可用且电网无限接纳。启用运行情景后，可按
月或任意自定义时段分别指定 **小时数、风资源、逐机可利用率、全场并网功率上限**，
电量按固定顺序形成五层口径：

```
毛发电 gross
  └ 扣 尾流损失 wake_loss
尾流后电量
  └ 扣 不可利用损失 availability_loss（冬季检修、逐机故障率）
可用电量
  └ 扣 限发损失 curtailment_loss（部分月份送出上限）
最终上网电量 delivered  → 交给经济分析（LCOE/NPV/IRR）
```

- 每个时段的小时数在尾流积分中恰好乘一次；时段权重只在汇总时放大已积分好的
  电量，二者不会重复计入。
- 限发按可解释规则在机组间分配：`proportional`（按可用功率同比例削减，默认）
  或 `merit_order`（按机组顺序优先保电）。
- 运行前统一校验并拒绝：缺失月份/小时数权重不闭合、负的或 >1 的可利用率、
  并网上限超过全场装机容量的冲突、非法 merit_order 等。
- **未配置时段（`operation.enabled=false`）时，AEP 与经济分析保持原值不变**，
  全可用月度情景复算旧流程，差异在数值精度以内。

用 `--operation` 指定情景 JSON（也可把 `operation` 块写进 `--config` 文件）：

```bash
python -m wind_farm_opt --operation operation_example.json --no-plots --output-dir output
```

### 月度模式（mode: monthly）

12 个自然月（非闰年，合计恰 8760 小时），`availability` 支持标量、长度 12 的
逐月标量或 `(12, N)` 的逐月逐机数组；`grid_capacity_mw` 支持 `null`/标量/逐月值。

```json
{
  "mode": "monthly",
  "availability": 1.0,
  "speed_factors": [1.15,1.12,1.05,1.0,0.92,0.85,0.82,0.83,0.9,1.02,1.1,1.14],
  "grid_capacity_mw": [null,null,null,null,null,null,null,null,null,null,30.0,null]
}
```

### 自定义时段模式（mode: custom）

任意时段，`hours*weight` 之和必须闭合到 8760（容差 ±1 小时）。逐月使用 weight=1；
"代表性时段"可用 weight 重复，此时 hours 不重复计入。

```json
{
  "mode": "custom",
  "curtailment_strategy": "merit_order",
  "periods": [
    {"name": "冬季检修", "hours": 2160, "wind_resource": {"speed_factor": 1.15},
     "availability": [0.7,1,1,1,1,1,1,1,1,1,1,1], "grid_capacity_mw": 35.0},
    {"name": "春秋", "hours": 4380, "availability": 1.0},
    {"name": "夏季小风", "hours": 2220, "wind_resource": {"speed_factor": 0.82},
     "availability": 1.0, "grid_capacity_mw": 30.0}
  ]
}
```

### 直接使用 Python API

```python
from wind_farm_opt.farm.operation import (
    build_monthly_scenario, ScenarioEnergyCalculator,
)

scenario = build_monthly_scenario(
    base_wind_resource, n_turbines,
    availability=availability_12xN,     # 逐机逐月
    grid_capacity_mw=[None]*11 + [30.0],
)
result = ScenarioEnergyCalculator(aep_calculator).compute(positions, scenario)
print(result.gross_mwh, result.wake_loss_mwh,
      result.availability_loss_mwh, result.curtailment_loss_mwh,
      result.delivered_mwh)
```

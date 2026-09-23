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

## 按月/自定义时段运行情景

默认流程把全年 8760 小时都视为机组完全可用、电网无限接纳。若要计入冬季检修、逐机故障率和部分月份的送出上限，可在配置文件中增加 `operation` 段，按"月或自定义时段"分别指定小时数、风资源、逐机可利用率和全场并网功率上限。

便捷的 12 个自然月写法（小时数自动取 744/672/…，合计恰为 8760）：

```json
"operation": {
  "total_hours": 8760,
  "require_full_coverage": true,
  "monthly": {
    "availability":  [0.90, 0.90, 0.97, 0.97, 0.97, 0.97,
                      0.97, 0.97, 0.97, 0.97, 0.97, 0.90],
    "grid_capacities_mw": [null, null, null, null, null, 20.0,
                           20.0, 20.0, null, null, null, null]
  }
}
```

也可以显式给出任意自定义时段（如检修窗口），每段可内联独立风资源：

```json
"operation": {
  "total_hours": 8760,
  "periods": [
    {"name": "冬季检修", "hours": 2160, "availability": 0.85},
    {"name": "春季",     "hours": 2208,
     "availability": 0.96, "grid_capacity_mw": null},
    {"name": "夏季送出受限", "hours": 2208, "grid_capacity_mw": 20.0},
    {"name": "秋季",     "hours": 2184,
     "wind_resource": {"type": "default",
                       "params": {"num_sectors": 12, "mean_speed": 9.0}}}
  ]
}
```

字段含义：

- `hours`：时段小时数。所有时段的小时数是唯一的时间权重，只在尾流/功率积分时进入一次，时段之间不重复计入。
- `availability`：逐机可利用率（0~1）；可传标量（全场相同）或长度等于机组台数的数组；缺省为 1。
- `grid_capacities_mw` / `grid_capacity_mw`：该时段全场并网功率上限（MW），缺省不限制。
- `wind_resource`：时段独立风资源，缺省使用项目主风资源。

计算严格按 **毛发电 → 尾流损失 → 不可利用损失 → 限发损失 → 最终上网电量** 的顺序逐级形成。限发在每个"风向×风速"运行状态上取 `min(全场可用功率, 并网上限)`，并按各机可用功率等比例（`proportional`）分配，规则可解释、可复现。结果中各损失彼此独立且相加闭合：

```
毛发电 − 尾流损失 − 不可利用损失 − 限发损失 = 最终上网电量
```

配置了运行情景时，经济分析（收益、LCOE、NPV、IRR）使用**最终上网电量**；`results.json` 的 `operation` 段给出全年与逐时段的五级电量分解。

下列问题会在**任何计算开始之前**被拒绝（抛出 `ScenarioValidationError`）：

- 缺失月份 / 各时段小时数之和与 `total_hours`（默认 8760）不闭合，或自定义时段小时数超过上限；
- 逐机可利用率为负、大于 1、含非有限值或长度与机组台数不符；
- 并网功率上限为负，或超过全场额定装机（容量上限冲突）；
- 时段为空、重名，或时段缺风资源且无默认风资源。

未配置 `operation`（或 `periods`/`monthly` 均为空）时，功能视为关闭，沿用原来的全年 8760 小时满发 AEP 流程，结果数值保持不变。运行情景相关逻辑见 `wind_farm_opt/farm/operation.py`，测试见 `tests/test_operation.py`（`python -m tests.test_operation`）。


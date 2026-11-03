# `mark-point` 分支：只读视觉查询

本分支从 main 独立派生，只增加 `mark_point`。它不包含另一个查询分支或 ICL 注入。
以前的 PhysLM `proposal` 同时带双视角框和主动视角点；本分支把其中一个能力拆成最小接口，
保留 sensor-only 几何语义，不声称旧 wire schema 逐字段兼容。

```json
{"stage":"mark_point","observation_id":"CURRENT","camera":"wrist","point_xy_px":[240,260]}
```

模型选择框/点，工具按原始图片像素绘制，不提供真值框或点。
严格读取所选像素的 optical depth，不做最近有效点搜索或深度吸附。回投影后返回参考系表面坐标与 TCP→表面的向量（mm）。7×7 邻域 p10/p50/p90 仅表示未分割深度分布；即使邻域有效，所选点无效时仍返回 geometry_valid=false。

## 接入

`Observation.query_views` 可选提供任意相机的 `QueryView`：

```python
from visual_query import QueryView
obs.query_views["wrist"] = QueryView(
    rgb=wrist_rgb,
    camera=wrist_camera_calibration,
    depth_optical_m=registered_optical_z_m,
)
```

深度必须是与最终显示 RGB 完全对齐的**光学轴 Z、米**；不是 uint16 mm、归一化
z-buffer、灰度深度图或欧氏距离。需要 adapter 先转换。有效范围默认 `0 < z < 5m`。
内参已包含镜像和裁剪，外参为 camera→reference，observation_id 和 frame 必须一致。
没有 depth 时仍可画标记，但 geometry_valid=false，不能推测三维坐标。

内置 wrist/external 图像可直接标记；额外相机由 query_views 提供。
每次观测默认最多 4 个成功查询，也消耗一次模型 call，但不消耗 motion/action/tick。
查询不 observe、不 step、不执行动作；返回标记图片+测量反馈，模型仍使用同一观测 ID。
查询图片归属于原观测组，随该组一起从最近 N 组图像历史中移除。
Native success 仍只由 benchmark 决定。

```bash
python examples/query_demo.py --output outputs/mark-point.png
# 只测试数学管线：使用人工 1m 平面；不能解释为真实场景深度。
python examples/query_demo.py --synthetic-depth --output outputs/mark-point-synthetic.png
python -m pytest -q
```

历史依据：PhysLM `physlm/rollout/bbox_geometry.py`、`sensor_motion.py` 的
`pixel_depth_feedback`、`interactive_traj.py` 的 `depth_feedback`（2026-11-03 核对）。
本分支未发起新 Kimi 请求或物理 rollout。

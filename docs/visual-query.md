# bbox-mark-point：同一分支中的框选与标点

本分支合并 bbox 和 mark-point。同一观测下两种查询都可用，也可以先框选再选点。
两者共享每次观测默认 4 次成功查询额度，各占一次模型 call，但不推进仿真、不消耗动作
或 tick。一条响应只调用一种查询；没有隐式框点联动。

## bbox

```json
{"stage":"bbox","observation_id":"CURRENT","camera":"wrist","bbox_xyxy_px":[120,180,300,350]}
```

整数原图坐标，半开区间 `[x0,x1) × [y0,y1)`。画出模型选的框。
有深度时回投影框内全部有效像素，分别返回 XYZ 的
min/p01/p05/p10/p50/p90/p95/p99/max，单位 mm。
没有物体分割或双目融合；桌面、背景、机器人和其他物体都可能进入统计。
不同轴的 p95 不是同一个三维点，也不是可直接执行的末端目标。

## mark_point

```json
{"stage":"mark_point","observation_id":"CURRENT","camera":"wrist","point_xy_px":[240,260]}
```

整数原图像素，画出模型选的点。有深度时严格读取那个像素，返回参考系表面坐标
及 TCP→表面向量，单位 mm；不自动吸附到邻近有效像素或修正深度。
7×7 邻域分位数只是未分割的深度分布提示，不能替代所选点的无效深度。
标点不自动受上一次 bbox 限制，也不会推断目标物体身份或抓取姿态。

## 共用输入与闭环

```python
from visual_query import QueryView
obs.query_views["wrist"] = QueryView(
    rgb=wrist_rgb,
    camera=wrist_camera_calibration,
    depth_optical_m=registered_optical_z_m,
)
```

深度是与最终 RGB 对齐的光学轴 Z、米，不是 uint16 mm、灰度深度、z-buffer 或
欧氏距离。有效范围默认 `0 < z < 5m`。内参包含镜像/裁剪；外参 camera→reference。
两种查询均核对当前 observation_id、相机参考系和图像尺寸。

未提供深度也能画框/点，但明确返回 geometry_valid=false，不生成三维结果。
查询返回标记图和数值反馈，不调用 observe/step/execute，不切换当前观测 ID。
两种查询图片归到同一观测组，随该组一起从最近 N 组历史中移除。
下一次物理动作后的新观测会重置共享额度。成功仍由 benchmark 原生判定。

内置 wrist/external 图像均可标注；其他相机由 query_views 显式提供。
每次查询图仅展示该次选择，不把旧标记累积成新的测量证据。

```bash
python examples/query_demo.py --query bbox --output outputs/bbox.png
python examples/query_demo.py --query mark_point --output outputs/point.png
# 可选数学管线测试：使用人工 1m 平面，不是真实场景深度。
python examples/query_demo.py --query mark_point --synthetic-depth --output outputs/point-plane.png
python -m pytest -q
```

ICL 仍单独放在 icl-demo 分支。历史语义来源是 PhysLM 的 bbox_geometry.py、
sensor_motion.py 和 interactive_traj.py，hash 见 query_source_hashes.json。
旧的 combined proposal wire 在这里拆成两个可选择的最小 schema。

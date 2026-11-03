# 原实验 harness 与适配边界

## 最新 LIBERO 两图配方

每次观察仅向模型提供 agentview 原始 RGB、带 TCP overlay 的 wrist RGB，加实测
机器人状态和上次动作反馈。深度被传感器记录保留，但没有作为图片输入 Kimi。
Demo 组固定保留训练成功演示的关键帧和 waypoint；实时图片保留最近 8 组，同时有
25,165,824 像素总预算。文本、返回的 reasoning 和数值反馈保留。

动作有三个：

1. `move_to`：相对当前**实测**末端的总平移和旋转增量。
   平移沿机器人基座轴；`R_target = R_delta @ R_current`。
2. `motion_traj`：基座系绝对 waypoint。位置线性插值，姿态 quaternion SLERP。
3. `finish`：结束尝试；成功仍由 benchmark 原生判定。

模型端位置 mm、axis-angle degrees（不是 Euler）；物理端 m、quaternion。
20 Hz，每段 1–60 ticks，夹爪每段 open/close/hold。不自动添加抓放宏动作、避障、
物理重试或 extra settle。较大实测跟踪误差中断动作并重新观察。

原批次预算为 120 calls / 40 motions / 2400 ticks / 12 h，输出上限 65536 tokens。
服务显式请求 `reasoning_effort=max`；后端是否兑现未经认证。此处是历史配置记录，
不是不同模型的通用参数。模型/档位/模板和预算应由新实验单独记录。

## RoboCasa mobile v3 的不同点

它使用 RGB-D + wrist overlay 共五图，而不是 LIBERO 两图配方。原生机器人为
PandaOmron 12D，参考系是**当前移动基座** `mobilebase0_base`；另有 `move_base`
和缓存的完成查询。归一化基座控制通道不是 mm 位移或 m/s。移动后需刷新相机和 TCP。

历史 RoboCasa 的 camera metadata 可能残留 `translation_frame=robot0_base` 标签；
真实矩阵使用移动基座，`coordinate_frame` / archived `axes_frame` 才是该 adapter 的
明确语义。公共 bridge 以 session 实测 base body 和显式 frame 为准。

## 几何接口

`T_reference_from_camera_optical = [R,t]` 表示 camera → reference，t 使用米。

```text
p_optical = Rᵀ (p_reference − t)
[u,v,1] ∝ K_image_px p_optical
axis_i(s) = measured_tcp_reference + s unit_axis_i
```

光学系 x 向右、y 向下、z 向前。K 已包含最终显示图像的 pixel center、resize、
crop、mirror。仅有 RGB 而缺少标定/机器人状态时，无法生成可信的厘米轴。
实机可由关节状态 + FK、手眼标定和已标定 TCP 提供这些输入。

## 现有 session 接入

```python
from tcp_overlay_adapter import render_session_overlays

session.observe()
paths, audit = render_session_overlays(
    session, output_dir,
    reference_frame="robot0_base",  # RoboCasa: mobilebase0_base
    views=("wrist",),
)
```

这是连接**已存在的 session**，不负责创建环境或安装 benchmark。其他仿真引擎可以
直接构造 `CameraCalibration`、`RobotGeometry` 调用 `render_tcp_overlay`，不使用
这个兼容 adapter。新 benchmark 的物理执行器必须单独验证坐标方向、控制频率、
旋转组合、实际到达误差和 native success。

## 最小闭环的职责

本仓库的 `harness.run` 在调用 policy 前生成图像，保留最近 N 组图片，将完整模型
输出作为审计事件；同一 observation_id 的合法动作最多执行一次。验证失败可以
让模型重新生成，物理执行抛异常则终止，不会当作“尚未执行”重试。

调用者负责 reset/close；policy 负责模型协议与网络 timeout；adapter 负责动作
schema/单位/边界/插值/控制/成功判定。新接口很小，且尚未在真实 LIBERO/RoboCasa
新 rollout 中验证；两个 benchmark 的真实投影回归只验证视觉层。

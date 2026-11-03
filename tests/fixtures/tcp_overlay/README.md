# TCP overlay projection fixtures

These are robot-only calibration/projection metadata copied on 2026-11-03 from
completed experiments. They contain no model credentials, object ground truth,
or generated trajectories. Expected pixels were produced by the historical
`tcp_axes_v2.py` implementations, before the common renderer was written.

- `libero.json`: `kimi_k3_libero40_twoimage_demo_ablation_pass5_20261102/`
  `demo/task-00-repeat-01/trial/visuals/ep125-node0-frame0/tcp_axes.json`.
- `robocasa.json`: `kimi_k3_robocasa111_mobile_v3_20261030/`
  `task-00-repeat-01/trial/visuals/ep20000-node0-frame0/tcp_axes.json`.

Both paths are relative to the project's existing shared `PhysLM/outputs/`.
LIBERO uses mirrored images (signed negative fx); RoboCasa uses unmirrored
images and the moving chassis frame. Both external and wrist projections are
tested. Image styling is intentionally not a byte-for-byte compatibility claim.

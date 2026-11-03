"""Draw a model-style selection on a real recorded frame; offline, no physics."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from harness import Observation  # noqa: E402
from tcp_overlay import CameraCalibration, RobotGeometry  # noqa: E402
from visual_query import QUERY_STAGE, QueryView, query  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--synthetic-depth", action="store_true",
                    help="Use a TEST constant 1m plane, never real recorded scene depth")
args = parser.parse_args()
source = Path(__file__).with_name("libero_snapshot.json")
snapshot = json.loads(source.read_text())
rgb = Image.open(source.parent / snapshot["image"]).convert("RGB")
cam, robot = CameraCalibration(**snapshot["camera"]), RobotGeometry(**snapshot["robot"])
obs = Observation(rgb, rgb, cam, robot, {})
if args.synthetic_depth:
    obs.query_views["wrist"] = QueryView(rgb, cam, np.ones((rgb.height, rgb.width)))
action = dict(stage=QUERY_STAGE, observation_id=robot.observation_id, camera="wrist")
action.update({"bbox_xyxy_px": [120, 180, 300, 350]} if QUERY_STAGE == "bbox"
              else {"point_xy_px": [240, 260]})
image, feedback = query(action, obs)
feedback["example_depth_source"] = "SYNTHETIC CONSTANT PLANE" if args.synthetic_depth else "none"
args.output.parent.mkdir(parents=True, exist_ok=True)
image.save(args.output)
args.output.with_suffix(".json").write_text(json.dumps(feedback, indent=2, allow_nan=False))
print(json.dumps({"stage": QUERY_STAGE, "geometry_valid": feedback["geometry_valid"],
                  "synthetic_depth": args.synthetic_depth, "executed_ticks": 0}))

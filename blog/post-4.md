---
layout: post
title: "Recreating FARM on Spot: From Fiducial to Scene Graph"
date: 2026-09-11
---

# Recreating FARM on Spot: From Fiducial to Scene Graph

*The setup notes behind [Replicating FARM in the Real World](/blog/post-3)
— before we could report on what works and what doesn't, we had to get the
pipeline running end-to-end on a real Spot. This is that part, written up
so it's repeatable next time (by us or anyone else).*

---

Getting FARM's scene-graph pipeline onto a physical Spot isn't really one
system — it's three, glued together: Boston Dynamics' GraphNav service
(which owns navigation and localization), a ROS 2 sensor stack (which owns
the RGB-D + odometry stream), and FARM's own offline scene-graph builder
(which turns that stream into a persistent 3D object map). None of the
three know about each other by default. The work is mostly in the seams.

## The one thing that actually matters: frame alignment

Before touching any commands, it's worth being explicit about the part
that will silently wreck the whole pipeline if you get it wrong.

A physical fiducial marker anchors both the navigation graph and the
robot's localization. When you start recording a nav graph, wherever Spot
happens to be standing becomes **waypoint 0** — the origin of its local
`odom` frame. GraphNav then reports a `seed_tform_body` transform at that
waypoint, which is the bridge from `odom` into the fiducial-anchored
**seed frame** — the frame the *entire* nav graph lives in.

The scene graph, meanwhile, is built directly from odometry, so every
object in it starts out expressed in `odom`. If you want the scene graph
and the nav graph to agree on where anything is — which is the entire
point of doing this — every object has to be carried through the same
chain:

```
scene-graph object (camera frame)
   ↓ T_cam→body
body frame
   ↓ T_body→odom       (from live odometry)
odom frame (waypoint 0 = (0,0))
   ↓ T_odom→seed        (from GraphNav's seed_tform_body at waypoint 0)
seed frame (shared with the nav graph)
```

Two practical consequences fall out of this: Spot has to be on the same
network as whatever machine is running the pipeline before any of this is
reachable, and **the fiducial's physical size has to be configured
correctly** — get that wrong and every downstream `seed` transform is
scaled and wrong in a way that's easy not to notice until objects and
waypoints stop lining up in the viewer.

## Step 1 — One Docker container, every session

Every capture session — whether you're recording a bag or streaming
live — starts from the same containerized ROS 2 environment:

```bash
docker run -it --rm --network=host \
    -v /dev:/dev \
    --privileged \
    --name scene_graph_nav_graph \
    --device-cgroup-rule="a *:* rmw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix -v ${XAUTH}:${XAUTH} \
    -e XAUTHORITY=${XAUTH} \
    --runtime nvidia --gpus all \
    -v ${PWD}:/workspace \
    -w=/workspace \
    -e LIBGL_ALWAYS_SOFTWARE="1" \
    -e DISPLAY=${DISPLAY} \
    spot_lio_sam:dev
```

Then, inside the container, bring up the Spot driver and an aligned
RealSense stream:

```bash
ros2 launch rover_launch spot_run.launch.py
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

This is the shared prerequisite for both paths below — record-then-process
(Step 3) or live streaming (Step 4).

## Step 2 — Record and upload the nav graph

Confirm Spot can be pinged and sits on the VLAN exposed by the edge
compute unit (an Orin, in our case). Power Spot on **at the spot that
should become waypoint 0**, then record with the Spot SDK's own tooling:

```bash
source ~/spot_sdk/bin/activate
python3 -m recording_command_line \
    --download-filepath ~/mist_ws_ros2/data_walk/nav_graph_spot_2 \
    192.168.50.3
```

Walk Spot through the space using the tool's interactive commands. Once
you're done, it's worth sanity-checking the waypoint-0 transform before
moving on — this is exactly the `T_odom→seed` from the diagram above:

```
waypoint_id: "lento-macaw-UIXPzApBFdLgdRhKVX0FuQ=="
waypoint_tform_body { position { x: -0.0202 y: -0.0142 z: 0.0090 } ... }
seed_tform_body {
  position { x: 2.2729 y: 0.1977 z: -0.5361 }
  rotation { x: 0.0048 y: 0.0038 z: 0.9995 w: 0.0300 }
}
```

Then upload the graph back onto Spot — required after any power cycle,
and best done with Spot physically near the fiducial so it can relocalize
immediately:

```bash
python3 -m graph_nav_command_line \
    --upload-filepath ~/mist_ws_ros2/scene_graph/data_walk/nav_graph_spot_4/downloaded_graph/ \
    192.168.50.3
```

## Step 3 — Record the rosbag

With the container from Step 1 running, record the synchronized RGB-D +
odometry stream you'll later turn into scene-graph frames:

```bash
ros2 bag record -o rosbags/scene_graph_spot_2 \
    /odometry \
    /camera/camera/color/image_raw \
    /camera/camera/depth/camera_info \
    /camera/camera/aligned_depth_to_color/camera_info \
    /camera/camera/aligned_depth_to_color/image_raw
```

Drive Spot through the space while this runs. This is the path we used
for the offline builds in the [previous post](/blog/post-3).

## Step 4 — Or: stream it live and teleop through Viser

As an alternative — or in addition — a small WebSocket bridge streams
Spot's video and odometry live and accepts navigation commands from a
Viser-based UI, driving Spot along the nav graph recorded in Step 2:

```bash
python ros_ws_bridge_2.py --host 0.0.0.0 --port 8765 \
    --odom-topic /odometry \
    --image-topic /camera/camera/color/image_raw/compressed \
    --image-qos best_effort --image-max-fps 10 --image-max-side 640

python3 spot_graphnav_goal.py \
    --robot-hostname 192.168.50.3 \
    --graph-path /workspace/scene_graph/data_walk/nav_graph_spot_4/downloaded_graph/ \
    --ws-url ws://127.0.0.1:8765
```

## Step 5 — Frames in, scene graph out

Once you have a bag, convert it into the `(rgb, depth, intrinsics, pose)`
tuples the offline builder expects:

```bash
python rgb_bag_frame.py \
    --bag-dir ../data/scene_graph_spot_6/ \
    --out-dir ../data/spot_scene_graph_frames_6 \
    --rgb-topic /camera/camera/color/image_raw \
    --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
    --camera-info-topic camera/camera/aligned_depth_to_color/camera_info \
    --odometry-topic /odometry
```

Then run the actual FARM-style builder — per-frame open-vocabulary
segmentation, 3D Gaussian fitting, cross-frame data association, and
(with `--caption`) async VLM captioning and multi-modal embedding, all
detailed in the [project write-up](/projects/projects_project-5):

```bash
python -m scene_graph.offline.run \
    --source frames-json \
    --frames-json-dir data/spot_scene_graph_frames_6 \
    --stride 5 \
    --save-path data/scene_graph_mist_lab_6_caption.pt \
    --covisibility \
    --caption
```

And visualize it, overlaying the nav graph from Step 2 as the practical
check that the frame-alignment work actually held:

```bash
python scripts/view_scene_state.py \
    --pt data/scene_graph_mist_lab_6_caption.pt \
    --walk nav_graph_spot_4/ \
    --grid-cell-m 1.0 \
    --no-camera-trajectory \
    --ws-url ws://192.168.1.192:8765
```

If the frame chain from the top of this post is right, objects and
waypoints should sit consistently in the same space. If they don't,
that's usually where the debugging starts.

## Where we're still stuck

A handful of things didn't get solved during this pass, and are the
active list for next time:

- **Odometry in the vision frame.** Getting odometry directly in the
  vision frame would remove the `cam_t_body` error currently baked into
  the pipeline — untried so far.
- **Camera recalibration.** The camera used for bag recording needs a
  proper recalibration pass.
- **An online version.** Everything above is record-then-process; a
  streaming version that skips the bag entirely is the natural next step,
  and ties into the live WS-bridge path in Step 4.
- **Walls and room boundaries.** We can't yet derive room extent or walls
  automatically — the visualizer's `--grid-cell-m` only sets a display
  grid, it doesn't constrain anything.
- **Navigable-area logic.** The flags that govern what counts as
  navigable space need more tuning than we've given them so far.

None of these blocked the demo runs in the [previous post](/blog/post-3),
but they're the reason "it works" comes with an asterisk. The code for
the scene-graph side is at
[GoldenGait/FARM-Project](https://github.com/GoldenGait/FARM-Project) if
you want to reproduce this yourself — `scene_graph.offline.run` is the
entry point either way.

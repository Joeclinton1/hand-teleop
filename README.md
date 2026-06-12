# hand-teleop

**hand-teleop** turns your webcam into real-time robot joint positions — no servers, just pure Python.

Designed for [LeRobot](https://github.com/huggingface/lerobot), it runs fast, smooth, and locally.

## Highlights

* ⚡ **Wilor model** (GPU) — accurate and real-time (currently the only fully working backend)
* 🧩 **AprilTag-based fingertip tracking** (experimental)
* 🖐️ Direct joint outputs — no inverse kinematics required
* 🔄 Built-in Kalman smoothing for low-jitter motion
* 🧵 Low-latency threading: tracking runs in the background
* 🧩 Plug-and-play with LeRobot

---

## Installation

### First Install the LeRobot fork!

This project is compatible with LeRobot but requires some changes to the teleoperation code to work. I have created a fork which uses it that should work with this library

https://github.com/Joeclinton1/lerobot/tree/hand_teleop

My fork is outdated now, and I am working to get it working with the newest verison of LeRobot but they changed the joint configuration in a way that is currently breaking this project. [Link to work in progress fork](https://github.com/Joeclinton1/lerobot/tree/hand_teleop_new)

### ✅ Recommended (Wilor – GPU-based, CUDA required)

```bash
pip install "hand-teleop @ git+https://github.com/joeclinton1/hand-teleop.git#egg=hand-teleop[wilor]"
```

This extra installs WiLoR from my fork:

```toml
wilor-mini = { git = "https://github.com/Joeclinton1/WiLoR-mini", optional = true }
```

The fork ports upstream WiLoR's fast inference path into WiLoR-mini. CUDA WiLoR uses this path by default in `hand-teleop`, enabling fp16 inference and the upstream ViT depth-pruning block skip.

> ✅ **Works well out of the box**
> ⚠️ **Requires GPU with CUDA**

---

### ⚠️ Experimental CPU-based Backends

```bash
pip install "hand-teleop @ git+https://github.com/joeclinton1/hand-teleop.git#egg=hand-teleop[mediapipe]"
pip install "hand-teleop @ git+https://github.com/joeclinton1/hand-teleop.git#egg=hand-teleop[apriltag]"
```

> 🧪 These are **almost working**, but not quite stable yet.
> 🙏 **PRs welcome** to help fix or improve them!

---

### 🛠 Development Install

```bash
conda create -n hand-teleop python=3.10 -y
conda activate hand-teleop
conda install -c conda-forge ffmpeg

git clone https://github.com/joeclinton1/hand-teleop.git
cd hand-teleop
pip install -e ".[wilor]"
```

> ⚠️ **Important:** If you're using `hand-teleop` alongside [LeRobot](https://github.com/huggingface/lerobot), it uses `opencv-python-headless`, which **breaks GUI functions** like `cv2.imshow()`.
>
> To fix this:
>
> ```bash
> pip uninstall opencv-python-headless opencv-python opencv-contrib-python
> pip install opencv-python
> ```
---

### Optional (for forward/inverse kinematics)

```bash
conda install -c conda-forge pinocchio
```

---

## AprilTag Setup (for cube-based tracking)

If you're experimenting with the `apriltag` model, here's the intended tag layout:

### Cube Tag Layout

Each cube is 2.5 cm with 1.8 cm-wide tags.

| Face   | Index Tags | Thumb Tags |
| ------ | ---------- | ---------- |
| Front  | 0          | 5          |
| Left   | 1          | 6          |
| Right  | 2          | 7          |
| Top    | 3          | 8          |
| Bottom | 4          | 9          |

```
      +------+       
      |  3   |     ↑ Top
 +----+------+----+
 |  1 |  0   |  2 |   → Front = 0
 +----+------+----+
      |  4   |     ↓ Bottom
      +------+
```

* Only **one visible tag per cube** is needed.
* Automatically selects the best visible tag.

---

### Assets

* STL: `assets/finger_tip_cubes.stl`
* Printable tags: `assets/tag25h9_0-9,0-9/`

---

## Basic Usage

```python
from hand_teleop import HandTracker

tracker = HandTracker(model="wilor", hand="right", cam_idx=0)
tracker.start()

while True:
    joints = tracker.get_joint_positions()
    robot.set_joint_positions(joints)
```

---

## Demo

```bash
python main.py
````

### Command-line options

* `--model wilor` — Hand model to use
* `--fps 30` — Frame rate (default: 60)
* `--quiet` — Silence console output
* `--no-joint` — Output raw gripper pose (pose-space mode)
* `--cam-idx 1` — Change camera index used for the tracking
* `--hand left` — Choose which hand to track (`left` or `right`, default: `right`)
* `--use-scroll` — Enable scroll-based gripper control

---

## LeRobot Integration

The package also exposes `DualHandTracker` for integrations that need both hands from one webcam/model pass. In the `gem` branch of my LeRobot fork, use `--teleop.type=hand_teleop` to drive a GEM-compatible leader action from hand tracking.

For a physical single GEM arm, use one hand:

```bash
lerobot-teleoperate --robot.type=gem --teleop.type=hand_teleop --teleop.hand=right
```

For the robot-arm viewer with two hands, use viewer-only bimanual mode:

```bash
lerobot-teleoperate --robot.type=gem --viewer.enabled=true --viewer.only=true --teleop.type=hand_teleop --teleop.hand=both
```

---

## Controls

* `p` — Pause/resume
* `space` — Hold to realign

---

## License

Apache 2.0

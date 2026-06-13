# ruff: noqa: N806 N803
"""Dual-hand tracker using one camera stream and one model forward per frame."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
from pynput import keyboard

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.gripper_pose.gripper_pose_computer import GripperPoseComputer
from hand_teleop.hand_pose.factory import ModelName, create_estimator
from hand_teleop.hand_pose.types import HandKeypointsPred
from hand_teleop.tracking.kalman_filter import KalmanXYZ
from hand_teleop.tracking.tracker import DEFAULT_CAM_T

HandName = Literal["left", "right"]


@dataclass
class _TrackedHandState:
    kf: KalmanXYZ
    kf_t: float
    prev_rel_pose: GripperPose
    initial_pose: Optional[GripperPose] = None
    base_pose: Optional[GripperPose] = None
    last_final_pose: Optional[GripperPose] = None


class DualHandPoseComputer:
    """Computes left/right hand poses from a shared hand-pose estimator call."""

    def __init__(self, device: Optional[str] = None, model: ModelName = "wilor"):
        self.estimator = create_estimator(model, device=device)
        self._pose_computer = GripperPoseComputer(device=device, model=model, estimator=self.estimator)

        self.robot_axes_in_hand = self._pose_computer.robot_axes_in_hand
        self.raw_abs_poses: dict[HandName, GripperPose] = {}

    def reset(self, hand: HandName | None = None) -> None:
        if hand is None:
            self.raw_abs_poses.clear()
        else:
            self.raw_abs_poses.pop(hand, None)

    def compute_absolute_poses(
        self, frame: np.ndarray, focal_length: float, cam_t: np.ndarray
    ) -> dict[HandName, GripperPose]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.estimator(frame_rgb, focal_length)
        poses: dict[HandName, GripperPose] = {}

        for pred in preds:
            hand = self._prediction_hand_after_frame_flip(pred)
            if hand in poses:
                continue
            pose = self._pose_computer._compute_gripper_pose(pred.keypoints)
            pose.change_basis(self.robot_axes_in_hand, cam_t)
            poses[hand] = pose

        self.raw_abs_poses = {hand: pose.copy() for hand, pose in poses.items()}
        return poses

    @staticmethod
    def _prediction_hand_after_frame_flip(pred: HandKeypointsPred) -> HandName:
        # The webcam frame is horizontally flipped before inference. Existing single-hand
        # code compensates by selecting the opposite classifier label.
        return "left" if pred.is_right else "right"


class DualHandTracker:
    def __init__(
        self,
        cam_idx: int = 0,
        device: Optional[str] = None,
        model: ModelName = "wilor",
        show_viz: bool = False,
        focal_ratio: float = 0.7,
        cam_t: np.ndarray = DEFAULT_CAM_T,
        urdf_path: Optional[str] = None,
        frame_name: str = "gripper_link",
        safe_range: Optional[dict[str, tuple[float, float]]] = None,
        debug_mode: bool = False,
        kf_dt: float = 1 / 30,
        kf_q: float = 5e-3,
        kf_r: float = 5e-3,
        start_paused: bool = False,
    ):
        self.focal_ratio = focal_ratio
        self.show_viz = show_viz
        self.cam_t = cam_t
        self.debug_mode = debug_mode
        self.safe_range = safe_range
        self._max_jump_rate = 2

        self.cap = cv2.VideoCapture(cam_idx)
        self.pose_computer = DualHandPoseComputer(device=device, model=model)
        self.robot_kin = (
            self._make_robot_kinematics(urdf_path, frame_name) if urdf_path is not None else None
        )

        now = time.perf_counter()
        self._hands: dict[HandName, _TrackedHandState] = {
            "left": _TrackedHandState(KalmanXYZ(dt=kf_dt, q=kf_q, r=kf_r), now, GripperPose.zero()),
            "right": _TrackedHandState(KalmanXYZ(dt=kf_dt, q=kf_q, r=kf_r), now, GripperPose.zero()),
        }
        self._lock = threading.Lock()
        self.tracking_paused = start_paused

        self._stop = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self) -> None:
        ema_fps = 60.0
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue

            frame = cv2.flip(frame, 1)
            if not self.tracking_paused:
                abs_poses = self.pose_computer.compute_absolute_poses(
                    frame,
                    self.focal_ratio * frame.shape[1],
                    self.cam_t,
                )
                for hand, abs_pose in abs_poses.items():
                    self._update_hand(hand, abs_pose)

            if self.show_viz:
                frame_time = time.perf_counter() - loop_start
                if frame_time > 1e-6:
                    ema_fps = 0.9 * ema_fps + 0.1 * (1.0 / frame_time)
                cv2.putText(
                    frame,
                    f"FPS: {ema_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'p' to pause | Hold SPACE to realign",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                )
                cv2.imshow("dual-hand-teleop", frame)
                cv2.waitKey(1)

    def _update_hand(self, hand: HandName, abs_pose: GripperPose) -> None:
        with self._lock:
            state = self._hands[hand]
            if state.initial_pose is None:
                state.initial_pose = abs_pose.copy()

            rel_pose = abs_pose.copy()
            rel_pose.inverse_transform_pose(state.initial_pose.rot, state.initial_pose.pos)

            now = time.perf_counter()
            dt = now - state.kf_t
            jump = np.linalg.norm(rel_pose.pos - state.prev_rel_pose.pos)
            max_jump = self._max_jump_rate * max(dt, 1e-4)
            if jump > max_jump:
                direction = rel_pose.pos - state.prev_rel_pose.pos
                rel_pose.pos = state.prev_rel_pose.pos + direction / jump * max_jump

            if dt > 0.0:
                state.kf.predict(dt)
            state.kf.update(rel_pose.pos)
            state.kf_t = now
            state.prev_rel_pose = rel_pose.copy()

    def _on_press(self, key):
        if key == keyboard.Key.space:
            self._pause()
        elif key == keyboard.KeyCode.from_char("p"):
            self._resume() if self.tracking_paused else self._pause()

    def _on_release(self, key):
        if key == keyboard.Key.space:
            self._resume()

    def _pause(self) -> None:
        self.tracking_paused = True
        with self._lock:
            for state in self._hands.values():
                state.kf.x[3:] = 0.0

    def _resume(self) -> None:
        self.tracking_paused = False
        with self._lock:
            now = time.perf_counter()
            for state in self._hands.values():
                state.kf.reset()
                state.kf_t = now
                state.prev_rel_pose = GripperPose.zero()
                state.initial_pose = None
                state.base_pose = None
        self.pose_computer.reset()

    def _predict_only(self, hand: HandName) -> None:
        if self.tracking_paused:
            return
        state = self._hands[hand]
        now = time.perf_counter()
        dt = now - state.kf_t
        if dt > 0.0:
            state.kf.predict(dt)
            state.kf_t = now

    def predict_pose(self, hand: HandName) -> GripperPose:
        with self._lock:
            self._predict_only(hand)
            state = self._hands[hand]
            pose = state.prev_rel_pose.copy()
            pose.pos = state.kf.x[:3]
            return pose

    def read_hand_state(self, hand: HandName, base_pose: GripperPose) -> GripperPose:
        rel = self.predict_pose(hand)
        with self._lock:
            state = self._hands[hand]
            if state.base_pose is None:
                state.base_pose = base_pose.copy()
            final_pose = state.base_pose.copy()
            final_pose.transform_pose(rel.rot, rel.pos)
            final_pose.open_degree = rel.open_degree
            state.last_final_pose = final_pose.copy()
            return final_pose

    def read_hand_state_joint(self, hand: HandName, base_pose_joint: np.ndarray) -> np.ndarray:
        if self.robot_kin is None:
            raise RuntimeError("robot_kin is not initialized. Pass a URDF to use this function.")

        arm_dof = self.robot_kin.nq
        if len(base_pose_joint) < arm_dof + 1:
            raise ValueError(
                f"Expected at least {arm_dof + 1} base joint values for {self.robot_kin.urdf_path}, "
                f"got {len(base_pose_joint)}."
            )

        arm_joints_rad = np.radians(base_pose_joint[:arm_dof])
        gripper_val = float(base_pose_joint[arm_dof])
        base_pose = self.robot_kin.fk(arm_joints_rad)
        base_gripper_pose = GripperPose.from_matrix(base_pose, open_degree=gripper_val)
        final_gripper_pose = self.read_hand_state(hand, base_gripper_pose)

        if self.safe_range:
            final_gripper_pose.clip(self.safe_range)

        new_arm_joints_rad = self.robot_kin.ik(
            arm_joints_rad.copy(), final_gripper_pose.to_matrix(), max_iters=6
        )
        new_arm_joints_deg = np.degrees(new_arm_joints_rad)
        return np.append(new_arm_joints_deg, final_gripper_pose.open_degree).astype(np.float32)

    def close(self) -> None:
        self._stop.set()
        self._listener.stop()
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _make_robot_kinematics(urdf_path: str, frame_name: str):
        from hand_teleop.kinematics.kinematics import RobotKinematics

        return RobotKinematics(urdf_path=urdf_path, frame_name=frame_name)

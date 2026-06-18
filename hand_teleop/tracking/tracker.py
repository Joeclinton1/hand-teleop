# ruff: noqa: N806 N803
"""
HandTracker
===========

• Owns the webcam loop **and** the Kalman filter.
• GripperPoseComputer is now a pure vision module.
• read_hand_state() returns a Kalman-smoothed pose for the API.
"""

import threading
import time
from typing import Literal, Optional

import cv2
import numpy as np
from pynput import keyboard, mouse

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.gripper_pose.gripper_pose_computer import GripperPoseComputer
from hand_teleop.gripper_pose.gripper_pose_visualizer import GripperPoseVisualizer
from hand_teleop.hand_pose.factory import ModelName
from hand_teleop.tracking.kalman_filter import KalmanXYZ

DEFAULT_CAM_T = np.array([0, -0.24, 0.6], dtype=np.float32)


class HandTracker:
    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        cam_idx: int = 0,
        device: Optional[str] = None,
        model: ModelName = "wilor",
        hand: Literal["left", "right"] = "right",
        show_viz: bool = True,
        focal_ratio: float = 0.7,
        cam_t: np.ndarray = DEFAULT_CAM_T,
        urdf_path: Optional[str] = None,
        frame_name: str = "gripper_link",
        use_scroll: bool = False,
        scroll_scale: float = -0.08,
        safe_range: Optional[dict[str, tuple[float, float]]] = None,
        debug_mode: bool = False,
        kf_dt: float = 1 / 30,
        kf_q: float = 5e-3,
        kf_r: float = 5e-3,
    ):
        # --- user options / visuals
        self.focal_ratio = focal_ratio
        self.show_viz = show_viz
        self.cam_t = cam_t
        self.debug_mode = debug_mode

        # --- webcam
        self.cap = cv2.VideoCapture(cam_idx)

        # --- cross-thread state ------------------------------------------------
        self.kf = KalmanXYZ(dt=kf_dt, q=kf_q, r=kf_r)
        self._kf_t = time.perf_counter()  # single shared timestamp
        self.prev_rel_pose = GripperPose.zero()  # last *relative* pose (for rot / grip)
        self.base_pose: Optional[GripperPose] = None
        self._last_final_pose: Optional[GripperPose] = None
        self._lock = threading.Lock()

        self.pose_computer = GripperPoseComputer(device=device, hand=hand, model=model)
        self.pose_visualizer = GripperPoseVisualizer(
            self.pose_computer.robot_axes_in_hand, self.focal_ratio, self.cam_t
        )

        self.robot_kin = (
            self._make_robot_kinematics(urdf_path, frame_name)
            if urdf_path is not None
            else None
        )

        self.safe_range = safe_range
        self._max_jump_rate = 2  # metres per second you’ll allow

        # --- scroll-wheel state --------------------------------------
        self.use_scroll = use_scroll
        self.scroll_scale = scroll_scale
        self._scroll_open = 0.0  # range [0, 90]

        # --- controls
        self.tracking_paused = True
        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.start()
        self.keypoints_only_mode = False

        if self.use_scroll:  # NEW
            self._scroll_listener = mouse.Listener(on_scroll=self._on_scroll)
            self._scroll_listener.start()

        # --- background loop
        self._stop = threading.Event()
        threading.Thread(target=self._capture_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Webcam / Kalman loop
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """
        Runs continuously in a daemon thread:
        • grabs a frame
        • gets a *relative* pose from the vision module (or None)
        • if we have a detection, predict-update the Kalman filter
        """
        ema_fps = 60.0

        while not self._stop.is_set():
            loop_start = time.perf_counter()

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue

            frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - self._kf_t if not self.tracking_paused else 0.0   # seconds
            # (leave self._kf_t unchanged for now)

            rel_pose = None
            if not self.tracking_paused:
                rel_pose = self.pose_computer.compute_relative_pose(
                    frame,
                    self.focal_ratio * frame.shape[1],
                    self.cam_t,
                )

                if rel_pose is not None:
                    jump = np.linalg.norm(rel_pose.pos - self.prev_rel_pose.pos)

                    max_jump = self._max_jump_rate * max(dt, 1e-4)  # avoid zero
                    if jump > max_jump:
                        direction = rel_pose.pos - self.prev_rel_pose.pos
                        rel_pose.pos = self.prev_rel_pose.pos + direction / jump * max_jump
                        if self.debug_mode:
                            print(
                                f"[WARN] Pose jump {jump:.3f} m capped to {max_jump:.3f} m "
                                f"(dt={dt:.3f}s)"
                            )

            # ------------------------------------------------------------------
            # Kalman: predict-update only when we have a measurement
            # ------------------------------------------------------------------
            if rel_pose is not None:
                with self._lock:
                    now = time.perf_counter()
                    dt = now - self._kf_t
                    if dt > 0.0:
                        self.kf.predict(dt)
                    self.kf.update(rel_pose.pos)
                    self._kf_t = now
                    self.prev_rel_pose = rel_pose.copy()

            # ------------------------------------------------------------------
            # optional viz
            # ------------------------------------------------------------------
            if self.show_viz:
                if (
                    rel_pose is not None
                    and self.pose_computer.raw_abs_pose is not None
                    and self._last_final_pose is not None
                    and not self.tracking_paused
                ):
                    frame = self.pose_visualizer.draw_all(
                        frame,
                        self.pose_computer.raw_abs_pose,
                        self._last_final_pose,
                        keypoints_only=self.keypoints_only_mode
                    )

                # --- Draw EMA FPS in top left ---
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

                # --- Draw help text in bottom left ---
                height = frame.shape[0]
                cv2.putText(
                    frame,
                    "Press 'p' to pause | Hold SPACE to realign",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                )

                cv2.imshow("hand-teleop", frame)
                cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Keyboard helpers
    # ------------------------------------------------------------------
    def _on_press(self, key):
        if key == keyboard.Key.space:
            self._pause()
        elif key == keyboard.KeyCode.from_char("p"):
            self._resume() if self.tracking_paused else self._pause()
        elif key == keyboard.KeyCode.from_char("k"):
            self.keypoints_only_mode = not self.keypoints_only_mode

    def _on_release(self, key):
        if key == keyboard.Key.space:
            self._resume()

    def _pause(self):
        self.tracking_paused = True
        with self._lock:
            self.kf.x[3:] = 0.0  # Zero velocity

    def _resume(self):
        self.tracking_paused = False
        with self._lock:
            self.kf.reset()
            self._kf_t = time.perf_counter()
            self.prev_rel_pose = GripperPose.zero()
        self.pose_computer.reset()
        self.base_pose = None

    def _on_scroll(self, x, y, dx, dy):
        # dy > 0  = scroll up  (open);  dy < 0 = scroll down (close)
        self._scroll_open = float(
            np.clip(self._scroll_open + dy * self.scroll_scale * 90, 0.0, 90.0)
        )

    @staticmethod
    def _make_robot_kinematics(urdf_path: str, frame_name: str):
        from hand_teleop.kinematics.kinematics import RobotKinematics

        return RobotKinematics(urdf_path=urdf_path, frame_name=frame_name)

    # ------------------------------------------------------------------
    # Internal helper – predict only
    # ------------------------------------------------------------------
    def _predict_only(self) -> None:
        if self.tracking_paused:
            return  # freeze everything during pause
        """Advance the filter to 'now' without using any new measurement."""
        now = time.perf_counter()
        dt = now - self._kf_t
        if dt > 0.0:
            self.kf.predict(dt)
            self._kf_t = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_pose(self) -> GripperPose:
        """
        Return the latest Kalman-filtered *relative* pose.
        """
        with self._lock:
            self._predict_only()
            pose = self.prev_rel_pose.copy()
            pose.pos = self.kf.x[:3]
            if self.use_scroll:
                pose.open_degree = self._scroll_open
            return pose

    def read_hand_state(self, base_pose: GripperPose) -> GripperPose:
        """
        Combine the current *relative* pose with the user-supplied
        baseline to get an absolute pose in the robot frame.
        Also stores the final pose for visualization.
        """
        rel = self.predict_pose()
        if self.base_pose is None:
            self.base_pose = base_pose.copy()

        final_pose = self.base_pose.copy()
        final_pose.transform_pose(rel.rot, rel.pos)
        final_pose.open_degree = rel.open_degree
        self._last_final_pose = final_pose.copy()  # Store for visualization
        return final_pose

    def read_hand_state_joint(self, base_pose_joint: np.ndarray) -> np.ndarray:
        """
        Compute the current absolute joint configuration by using a joint-space base pose
        and applying the relative hand motion as a delta in cartesian space.

        Expects and returns a 6-element array in degrees: [j1, j2, j3, j4, j5, gripper_open].
        """
        if self.robot_kin is None:
            raise RuntimeError(
                "robot_kin is not initialized. Pass a URDF to use this function."
            )

        arm_dof = self.robot_kin.nq
        if len(base_pose_joint) < arm_dof + 1:
            raise ValueError(
                f"Expected at least {arm_dof + 1} base joint values for {self.robot_kin.urdf_path}, "
                f"got {len(base_pose_joint)}."
            )

        # Convert base joint angles to radians for kinematics.
        arm_joints_rad = np.radians(base_pose_joint[:arm_dof])
        gripper_val = float(base_pose_joint[arm_dof])  # gripper remains in degrees

        # Forward kinematics in radians
        base_pose = self.robot_kin.fk(arm_joints_rad)
        base_gripper_pose = GripperPose.from_matrix(base_pose, open_degree=gripper_val)

        # Apply relative hand motion
        final_gripper_pose = self.read_hand_state(base_gripper_pose)

        if self.safe_range:
            final_gripper_pose.clip(self.safe_range)

        # Inverse kinematics returns radians
        new_arm_joints_rad = self.robot_kin.ik(
            arm_joints_rad.copy(), final_gripper_pose.to_matrix(), max_iters=6
        )

        # Convert result back to degrees
        new_arm_joints_deg = np.degrees(new_arm_joints_rad)

        if self.debug_mode:
            print("Pose:", final_gripper_pose.to_string())

        return np.append(new_arm_joints_deg, final_gripper_pose.open_degree).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Cleanup helper (optional)
    # ------------------------------------------------------------------
    def close(self):
        self._stop.set()
        if self.use_scroll:
            self._scroll_listener.stop()
        self.cap.release()
        cv2.destroyAllWindows()

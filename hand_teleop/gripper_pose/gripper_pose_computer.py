# ruff: noqa: N806 N803

from typing import Literal, Optional

import cv2
import numpy as np

from hand_teleop.gripper_pose.gripper_pose import GripperPose
from hand_teleop.hand_pose.factory import (
    ModelName,
    create_estimator,
)
from hand_teleop.hand_pose.estimators.base import HandPoseEstimator
from hand_teleop.hand_pose.types import TrackedHandKeypoints


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


class GripperPoseComputer:
    GRIP_ANGLE_OFFSET = -2  # degrees

    def __init__(
        self,
        device: Optional[str] = None,
        model: ModelName = "wilor",
        hand: Literal["left", "right"] = "right",
        estimator: Optional[HandPoseEstimator] = None,
    ):
        self.estimator = estimator or create_estimator(model, device=device)
        self.hand = hand

        self.initial_pose: Optional[GripperPose] = None
        self.raw_abs_pose: Optional[GripperPose] = None  # for visualization

        self.robot_axes_in_hand = np.column_stack([
            [0, 0, -1],   # new x-axis
            [-1, 0, 0],   # new y-axis
            [0, 1, 0]     # new z-axis
        ])
        self.R_hand_to_robot = self.robot_axes_in_hand.T

    def reset(self):
        self.initial_pose = None

    def compute_relative_pose(
        self, frame: np.ndarray, focal_length: float, cam_t: np.ndarray
    ) -> Optional[GripperPose]:
        """
        Returns the relative pose (translation from initial frame).
        Orientation and gripper angle are raw (not relative).
        """
        abs_pose = self._get_absolute_pose(frame, focal_length, cam_t)
        if abs_pose is None:
            return None

        self.raw_abs_pose = abs_pose.copy()

        if self.initial_pose is None:
            self.initial_pose = abs_pose.copy()

        rel_pose = abs_pose.copy()
        rel_pose.inverse_transform_pose(
            self.initial_pose.rot,
            self.initial_pose.pos
        )
        return rel_pose

    def _get_absolute_pose(
        self, frame: np.ndarray, focal_length: float, cam_t: np.ndarray
    ) -> Optional[GripperPose]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.estimator(frame_rgb, focal_length)
        if len(preds)>1:
            corrected_hand = "left" if self.hand == "right" else "right" # we do this because we use expect the frame to be flipped horizontally
            preds = [p for p in preds if p.is_right == (corrected_hand == "right")]

        if not preds:
            return None

        keypoints = preds[0].keypoints
        pose = self._compute_gripper_pose(keypoints)
        pose.change_basis(self.robot_axes_in_hand, cam_t)
        return pose

    def _compute_gripper_pose(self, kp: TrackedHandKeypoints) -> GripperPose:
        # Coordinate axes from keypoints
        x_raw = normalize(kp.middle_base - kp.index_base)
        y_axis = normalize(kp.index_base - kp.thumb_mcp)
        z_axis = normalize(np.cross(x_raw, y_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
        R_hand = np.column_stack([x_axis, y_axis, z_axis])

        origin = 0.5 * (kp.index_base + kp.middle_base)
        tip = 0.5 * (kp.index_tip + kp.middle_tip)
        vec1 = tip - origin

        # Compute grip angle
        plane_n = x_axis
        thumb_root = kp.thumb_mcp - np.dot(kp.thumb_mcp - origin, plane_n) * plane_n
        thumb_tip_proj = kp.thumb_tip - np.dot(kp.thumb_tip - origin, plane_n) * plane_n
        vec2 = normalize(thumb_tip_proj - thumb_root) * np.linalg.norm(vec1)
        vec3 = normalize(thumb_tip_proj - origin) * np.linalg.norm(vec1)

        closed_angle = np.degrees(
            np.arccos(np.clip(np.dot(normalize(vec3), normalize(vec2)), -1, 1))
        ) * np.sign(np.dot(np.cross(vec2, vec3), plane_n))

        angle_rad = np.arccos(np.clip(np.dot(normalize(vec2), normalize(vec1)), -1.0, 1.0))
        grip_angle = (
            np.degrees(angle_rad)
            * np.sign(np.dot(np.cross(vec2, vec1), plane_n))
            - closed_angle
            + self.GRIP_ANGLE_OFFSET
        )

        return GripperPose(
            origin,
            R_hand,
            grip_angle,
            [origin.copy(), tip, thumb_root, thumb_tip_proj],
        )

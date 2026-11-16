"""Pelvis displacement and turn of executed joints (Z-up half-physics frame), used by the log summary of a try."""
import numpy as np

from ..world.hp_bridge import heading


def frame_axes(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])


def motion_summary(J, orient):
    """J: (T,22,3) Z-up joints, orient: (T,3) global orient -> displacement (forward, left) in the start frame and turn (deg)."""
    yaw = np.unwrap([heading(o) for o in orient]); f, l = frame_axes(yaw[0]); d = J[-1, 0, :2] - J[0, 0, :2]
    return float(d @ f), float(d @ l), float(np.degrees(yaw[-1] - yaw[0]))

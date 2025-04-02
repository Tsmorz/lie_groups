"""Add a doc string to my files."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy import linalg
from scipy.spatial.transform import Rotation as Rot

from lie_groups_py.definitions import EULER_ORDER, VECTOR_LENGTH


class SO3:
    """Represent a three-dimensional pose."""

    def __init__(
        self,
        yaw_pitch_roll: Optional[tuple[float, float, float]] = None,
        rot: Optional[np.ndarray] = None,
    ):
        if isinstance(yaw_pitch_roll, tuple | np.ndarray):
            rot = Rot.from_euler(angles=yaw_pitch_roll, seq=EULER_ORDER)
            self.rot = rot.as_matrix()
        elif isinstance(rot, np.ndarray):
            self.rot = rot
        else:
            msg = "Either 'roll_pitch_yaw' or 'rot' must be provided."
            logger.error(msg)
            raise ValueError(msg)

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        yaw, pitch, roll = self.as_vector()
        msg = f"SO3 Pose=(yaw:{float(yaw):.2f}, pitch:{float(pitch):.2f}, roll:{float(roll):.2f})"
        return msg

    def __matmul__(self, other):
        """Perform a matrix multiplication between two SE3 matrices."""
        if isinstance(other, SO3):
            new_se3 = self.as_matrix() @ other.as_matrix()
            return SO3(rot=new_se3)
        else:
            msg = "Matrix multiplication is only supported between SO3 poses."
            logger.error(msg)
            raise TypeError(msg)

    def as_vector(self, degrees: bool = False) -> np.ndarray:
        """Represent the data as a 3-by-1 matrix."""
        rot = Rot.from_matrix(matrix=self.rot)
        ypr = rot.as_euler(EULER_ORDER, degrees=degrees)
        return np.reshape(ypr, (3, 1))

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        return self.rot

    def inv(self) -> "SO3":
        """Return the inverse of the SO3 pose."""
        return SO3(rot=self.rot.T)

    def plot(
        self,
        ax,
        vector_length: float = VECTOR_LENGTH,
        xyz: Optional[tuple[float, float, float]] = None,
    ) -> None:  # pragma: no cover
        """Plot the pose in 3D space.

        :param ax: The axis to plot the pose.
        :param vector_length: The length of the vectors representing the pose axes.
        :param xyz: The 3D translation vector to plot the pose at. Defaults to (0, 0, 0).
        :return: None
        """
        if xyz is None:
            x, y, z = (0.0, 0.0, 0.0)
        else:
            x, y, z = xyz
        for i, color in enumerate(["r", "g", "b"]):
            u, v, w = vector_length * self.rot[i, :]
            ax.quiver(X=x, Y=y, Z=z, U=u, V=v, W=w, color=color)


def interpolate_so3(pose_0: SO3, pose_1: SO3, t: float | np.floating) -> SO3:
    """Interpolate between two SO3 poses.

    :param pose_0: The first SO3 pose.
    :param pose_1: The second SO3 pose.
    :param t: The interpolation parameter.
    :return: The interpolated SO3 pose.
    """
    rot_0, rot_1 = pose_0.rot, pose_1.rot
    s = linalg.logm(rot_1 @ rot_0.T)
    expt_s = linalg.expm(t * s)
    rot_new = expt_s @ rot_0
    return SO3(rot=rot_new)

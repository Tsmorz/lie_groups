"""Add a doc string to my files."""

from typing import Optional

import numpy as np
from loguru import logger
from scipy import linalg
from scipy.spatial.transform import Rotation as Rot

from se3_group.definitions import EULER_ORDER, VECTOR_LENGTH


class SE3:
    """Represent a three-dimensional pose."""

    def __init__(
        self,
        xyz: Optional[np.ndarray] = None,
        roll_pitch_yaw: Optional[np.ndarray] = None,
        rot: Optional[np.ndarray] = None,
    ):
        if xyz is None:
            msg = "The translations vector 'xyz' must be provided."
            logger.error(msg)
            raise ValueError(msg)
        if xyz.shape == (3,):
            xyz = np.reshape(xyz, (3, 1))

        if rot is not None:
            self.rot = rot
        elif roll_pitch_yaw is not None:
            if roll_pitch_yaw.shape == (3, 1):
                roll_pitch_yaw = np.reshape(roll_pitch_yaw, (3,))
            rot = Rot.from_euler(angles=roll_pitch_yaw, seq=EULER_ORDER)
            self.rot = rot.as_matrix()
        else:
            msg = "Either 'roll_pitch_yaw' or 'rot' must be provided."
            logger.error(msg)
            raise ValueError(msg)

        self.trans = xyz

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        x, y, z, roll, pitch, yaw = self.as_vector()
        msg = (
            f"SE3 Pose=(x:{float(x):.2f}, y:{float(y):.2f}, z:{float(z):.2f}, "
            f"roll:{float(roll):.2f}, pitch:{float(pitch):.2f}, yaw:{float(yaw):.2f})"
        )
        return msg

    def __matmul__(self, other):
        """Perform a matrix multiplication between two SE3 matrices."""
        if isinstance(other, SE3):
            new_se3 = self.as_matrix() @ other.as_matrix()
            return SE3(xyz=new_se3[:3, -1], rot=new_se3[:3, :3])
        else:
            msg = "Matrix multiplication is only supported between SE3 poses."
            logger.error(msg)
            raise ValueError(msg)

    def as_vector(self, degrees: bool = False) -> np.ndarray:
        """Represent the data as a 6-by-1 matrix."""
        xyz = np.reshape(self.trans, (3,))
        rot = Rot.from_matrix(matrix=self.rot)
        rpy = rot.as_euler(EULER_ORDER, degrees=degrees)
        return np.hstack((xyz, rpy))

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        matrix = np.hstack((self.rot, self.trans))
        matrix = np.vstack((matrix, np.array([[0.0, 0.0, 0.0, 1.0]])))
        return matrix

    def inv(self) -> "SE3":
        """Return the inverse of the SE3 pose."""
        rot_inv = np.linalg.inv(self.rot)
        trans_inv = -rot_inv @ self.trans
        return SE3(rot=rot_inv, xyz=trans_inv)

    def plot(self, ax, vector_length: float = VECTOR_LENGTH) -> None:
        """Plot the pose in 3D space.

        :param ax: The axis to plot the pose.
        :param vector_length: The length of the vectors representing the pose axes.
        :return: None
        """
        x, y, z = self.trans
        for i, color in enumerate(["r", "g", "b"]):
            u, v, w = vector_length * self.rot[i, :]
            ax.quiver(X=x, Y=y, Z=z, U=u, V=v, W=w, color=color)


def interpolate(pose_0: SE3, pose_1: SE3, t: float | np.floating) -> SE3:
    """Interpolate between two SE3 poses.

    :param pose_0: The first SE3 pose.
    :param pose_1: The second SE3 pose.
    :param t: The interpolation parameter.
    :return: The interpolated SE3 pose.
    """

    def vt(S: np.ndarray, t: float) -> np.ndarray:
        theta = np.linalg.norm([S[0, 1], S[0, 1], S[1, 2]])
        v = (
            np.eye(3)
            + (1 - np.cos(t * theta)) / (t * theta) ** 2 * t * S
            + (t * theta - np.sin(t * theta)) / (t * theta) ** 3 * t**2 * S @ S
        )
        return v

    if t == 0.0:
        return pose_0

    rot_0, tran_0 = pose_0.rot, pose_0.trans
    rot_1, tran_1 = pose_1.rot, pose_1.trans
    rot_new = rot_1 @ rot_0.T
    t_new = tran_1 - rot_new @ tran_0

    s = linalg.logm(rot_new)
    u = linalg.inv(vt(s, 1.0)) @ t_new
    expt_s = linalg.expm(t * s)

    rpy = Rot.from_matrix(matrix=expt_s @ rot_0).as_euler(EULER_ORDER, degrees=False)
    xyz = expt_s @ tran_0 + t * vt(s, t) @ u
    return SE3(xyz=xyz, roll_pitch_yaw=rpy)

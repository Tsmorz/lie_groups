"""Add a doc string to my files."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from lie_groups_py.definitions import VECTOR_LENGTH


class SE2:
    """Represent a three-dimensional pose."""

    def __init__(
        self,
        xy: tuple[float, float] | np.ndarray,
        yaw: Optional[float] = None,
        rot: Optional[np.ndarray] = None,
    ):
        if isinstance(yaw, float):
            self.rot = np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            )
        elif isinstance(rot, np.ndarray):
            self.rot = rot
        else:
            msg = "Either 'yaw' or 'rot' must be provided."
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(xy, np.ndarray):
            xy = np.reshape(xy, (2, 1))

        if isinstance(xy, tuple):
            x, y = xy
            xy = np.array([[x], [y]])

        self.trans = xy

    def __str__(self):  # pragma: no cover
        """Return a string representation of the pose."""
        x, y, yaw = self.as_vector()
        msg = f"SE2 Pose=(x:{float(x):.2f}, y:{float(y):.2f}, yaw:{float(yaw):.2f})"
        return msg

    def __matmul__(self, other):
        """Perform a matrix multiplication between two SE3 matrices."""
        if isinstance(other, SE2):
            new_se2 = self.as_matrix() @ other.as_matrix()
            return SE2(xy=new_se2[:2, -1], rot=new_se2[:2, :2])
        else:
            msg = "Matrix multiplication is only supported between SE2 poses."
            logger.error(msg)
            raise TypeError(msg)

    def as_vector(self) -> np.ndarray:
        """Represent the data as a 6-by-1 matrix."""
        x, y = np.reshape(self.trans, (2,))
        yaw = np.acos(self.rot[0, 0])
        return np.array([[x], [y], [yaw]])

    def as_matrix(self) -> np.ndarray:
        """Represent the data as a 3-by-3 matrix."""
        matrix = np.hstack((self.rot, self.trans))
        matrix = np.vstack((matrix, np.array([[0.0, 0.0, 1.0]])))
        return matrix

    def inv(self) -> "SE2":
        """Return the inverse of the SE3 pose."""
        rot_inv = np.linalg.inv(self.rot)
        trans_inv = -rot_inv @ self.trans
        return SE2(rot=rot_inv, xy=trans_inv)

    def plot(
        self, ax: plt.axes, vector_length: float = VECTOR_LENGTH
    ) -> None:  # pragma: no cover
        """Plot the pose in 2D space.

        :param ax: The axis to plot the pose.
        :param vector_length: The length of the vectors representing the pose axes.
        :return: None
        """
        x, y = self.trans
        theta = np.acos(self.rot[0, 0])
        dx = np.cos(theta) * vector_length
        dy = np.sin(theta) * vector_length
        ax.arrow(x=x, y=y, dx=dx, dy=dy)

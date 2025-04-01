"""Basic docstring for my module."""

import numpy as np
import pytest

from lie_groups_py.se2 import SE2


def test_se2_with_xy_1d() -> None:
    """Assert that the se2 pose is created correctly with 1D array for xy."""
    # Arrange
    xy = np.zeros(2)
    yaw = 0.0

    # Act
    se2_pose = SE2(xy=xy, yaw=yaw)

    # Assert
    np.testing.assert_array_almost_equal(se2_pose.as_matrix(), np.eye(3))
    np.testing.assert_array_almost_equal(se2_pose.as_vector(), np.zeros((3, 1)))


def test_se2_with_yaw() -> None:
    """Assert that the se2 pose is created correctly with a yaw angle."""
    # Arrange
    xy = (0.0, 0.0)
    yaw = 0.0

    # Act
    se2_pose = SE2(xy=xy, yaw=yaw)

    # Assert
    np.testing.assert_array_almost_equal(se2_pose.as_matrix(), np.eye(3))
    np.testing.assert_array_almost_equal(se2_pose.as_vector(), np.zeros((3, 1)))


def test_se2_with_rot() -> None:
    """Assert that the se2 pose is created correctly with a 2x2 rotation matrix."""
    # Arrange
    xy = (0.0, 0.0)
    rot = np.eye(2)

    # Act
    se2_pose = SE2(xy=xy, rot=rot)

    # Assert
    np.testing.assert_array_almost_equal(se2_pose.as_matrix(), np.eye(3))
    np.testing.assert_array_almost_equal(se2_pose.as_vector(), np.zeros((3, 1)))


def test_se2_missing_orientation() -> None:
    """Assert that a ValueError is raised when orientation is missing."""
    # Arrange
    xy = (0.0, 0.0)

    # Act and Assert
    with pytest.raises(ValueError):
        SE2(xy=xy)


def test_se2_multiplication() -> None:
    """Assert that the multiplication of two SE2 poses is correct."""
    # Arrange
    pose_0 = SE2(xy=(0.0, 0.0), yaw=0.0)
    pose_1 = SE2(xy=(0.0, 0.0), yaw=np.pi / 2)

    # Act
    new_pose = pose_0 @ pose_1

    # Assert
    np.testing.assert_array_almost_equal(pose_1.as_matrix(), new_pose.as_matrix())


def test_se2_multiplication_wrong_type() -> None:
    """Assert that a TypeError is raised when multiplying a SE2 pose with a non-SE2 pose."""
    # Arrange
    pose_0 = SE2(xy=(0.0, 0.0), yaw=0.0)
    pose_1 = SE2(xy=(0.0, 0.0), yaw=np.pi / 2)

    # Assert and Assert
    with pytest.raises(TypeError):
        pose_0 @ pose_1.as_matrix()


def test_se2_inverse() -> None:
    """Assert that the inverse of a SE2 pose is correct."""
    # Arrange
    pose_0 = SE2(xy=(0.0, 0.0), yaw=0.0)

    # Act
    inv_pose = pose_0.inv()

    # Assert
    np.testing.assert_array_almost_equal(pose_0.as_matrix(), inv_pose.as_matrix())

"""Basic docstring for my module."""

import numpy as np
import pytest

from lie_groups_py.se3 import SE3, interpolate_se3


def test_se3_with_xyz_1d() -> None:
    """Assert that the se3 pose is created correctly with 1D array for xyz."""
    # Arrange
    xyz = np.zeros(3)
    roll_pitch_yaw = (0.0, 0.0, 0.0)

    # Act
    se3_pose = SE3(xyz=xyz, yaw_pitch_roll=roll_pitch_yaw)

    # Assert
    np.testing.assert_array_almost_equal(se3_pose.as_matrix(), np.eye(4))
    np.testing.assert_array_almost_equal(se3_pose.as_vector(), np.zeros((6, 1)))


def test_se3_with_rpy() -> None:
    """Assert that the se3 pose is created correctly with RPY angles."""
    # Arrange
    xyz = (0.0, 0.0, 0.0)
    roll_pitch_yaw = (0.0, 0.0, 0.0)

    # Act
    se3_pose = SE3(xyz=xyz, yaw_pitch_roll=roll_pitch_yaw)

    # Assert
    np.testing.assert_array_almost_equal(se3_pose.as_matrix(), np.eye(4))
    np.testing.assert_array_almost_equal(se3_pose.as_vector(), np.zeros((6, 1)))


def test_se3_with_rot() -> None:
    """Assert that the se3 pose is created correctly with a 3x3 rotation matrix."""
    # Arrange
    xyz = (0.0, 0.0, 0.0)
    rot = np.eye(3)

    # Act
    se3_pose = SE3(xyz=xyz, rot=rot)

    # Assert
    np.testing.assert_array_almost_equal(se3_pose.as_matrix(), np.eye(4))
    np.testing.assert_array_almost_equal(se3_pose.as_vector(), np.zeros((6, 1)))


def test_se3_missing_orientation() -> None:
    """Assert that a ValueError is raised when orientation is missing."""
    # Arrange
    xyz = (0.0, 0.0, 0.0)

    # Act and Assert
    with pytest.raises(ValueError):
        SE3(xyz=xyz)


def test_se3_multiplication() -> None:
    """Assert that the multiplication of two SE3 poses is correct."""
    # Arrange
    pose_0 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(0.0, 0.0, 0.0))
    pose_1 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Act
    new_pose = pose_0 @ pose_1

    # Assert
    np.testing.assert_array_almost_equal(pose_1.as_matrix(), new_pose.as_matrix())


def test_se3_multiplication_wrong_type() -> None:
    """Assert that a TypeError is raised when multiplying a SE3 pose with a non-SE3 pose."""
    # Arrange
    pose_0 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(0.0, 0.0, 0.0))
    pose_1 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Assert and Assert
    with pytest.raises(TypeError):
        pose_0 @ pose_1.as_matrix()


def test_se3_inverse() -> None:
    """Assert that the inverse of a SE3 pose is correct."""
    # Arrange
    pose_0 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(0.0, 0.0, 0.0))

    # Act
    inv_pose = pose_0.inv()

    # Assert
    np.testing.assert_array_almost_equal(pose_0.as_matrix(), inv_pose.as_matrix())


def test_interpolate_se3() -> None:
    """Assert that the interpolated se3 poses are correct."""
    # Arrange
    se3_0 = SE3(xyz=(0.0, 0.0, 0.0), yaw_pitch_roll=(0.0, 0.0, 0.0))
    se3_1 = SE3(xyz=(2.0, 4.0, 8.0), yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Act
    se3_new_t0 = interpolate_se3(se3_0, se3_1, 0.0)
    se3_new_t1 = interpolate_se3(se3_0, se3_1, 1.0)

    # Assert
    np.testing.assert_array_almost_equal(se3_0.as_matrix(), se3_new_t0.as_matrix())
    np.testing.assert_array_almost_equal(se3_1.as_matrix(), se3_new_t1.as_matrix())

"""Basic docstring for my module."""

import numpy as np
import pytest

from lie_groups_py.so3 import SO3, interpolate_so3


def test_so3_ypr() -> None:
    """Assert that the se3 pose is created correctly with 1D array for xyz."""
    # Arrange
    yaw_pitch_roll = (0.0, 0.0, 0.0)

    # Act
    so3_pose = SO3(yaw_pitch_roll=yaw_pitch_roll)

    # Assert
    np.testing.assert_array_almost_equal(so3_pose.as_matrix(), np.eye(3))
    np.testing.assert_array_almost_equal(so3_pose.as_vector(), np.zeros((3, 1)))


def test_so3_with_rot() -> None:
    """Assert that the so3 pose is created correctly with RPY angles."""
    # Arrange
    rot = np.eye(3)

    # Act
    so3_pose = SO3(rot=rot)

    # Assert
    np.testing.assert_array_almost_equal(so3_pose.as_matrix(), np.eye(3))


def test_so3_missing_orientation() -> None:
    """Assert that a ValueError is raised when orientation is missing."""
    # Act and Assert
    with pytest.raises(ValueError):
        SO3()


def test_so3_multiplication() -> None:
    """Assert that the multiplication of two SE3 poses is correct."""
    # Arrange
    pose_0 = SO3(yaw_pitch_roll=(0.0, 0.0, 0.0))
    pose_1 = SO3(yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Act
    new_pose = pose_0 @ pose_1

    # Assert
    np.testing.assert_array_almost_equal(pose_1.as_matrix(), new_pose.as_matrix())


def test_so3_multiplication_wrong_type() -> None:
    """Assert that a TypeError is raised when multiplying a SE3 pose with a non-SE3 pose."""
    # Arrange
    pose_0 = SO3(yaw_pitch_roll=(0.0, 0.0, 0.0))
    pose_1 = SO3(yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Assert and Assert
    with pytest.raises(TypeError):
        pose_0 @ pose_1.as_matrix()


def test_so3_inverse() -> None:
    """Assert that the inverse of a SE3 pose is correct."""
    # Arrange
    pose_0 = SO3(yaw_pitch_roll=(np.pi / 2, 0.0, 0.0))

    # Act
    inv_pose = pose_0.inv()

    # Assert
    np.testing.assert_array_almost_equal(pose_0.as_matrix().T, inv_pose.as_matrix())


def test_interpolate_so3() -> None:
    """Assert that the interpolated se3 poses are correct."""
    # Arrange
    se3_0 = SO3(yaw_pitch_roll=(0.0, 0.0, 0.0))
    se3_1 = SO3(yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8))

    # Act
    se3_new_t0 = interpolate_so3(se3_0, se3_1, t=0.0)
    se3_new_t1 = interpolate_so3(se3_0, se3_1, t=1.0)

    # Assert
    np.testing.assert_array_almost_equal(se3_0.as_matrix(), se3_new_t0.as_matrix())
    np.testing.assert_array_almost_equal(se3_1.as_matrix(), se3_new_t1.as_matrix())

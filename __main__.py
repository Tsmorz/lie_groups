"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from lie_groups_py.se3 import SE3, interpolate_se3


def main() -> None:
    """Run a simple demonstration."""
    pose_0 = SE3(
        xyz=np.array([0.0, 0.0, 0.0]),
        rot=np.eye(3),
    )
    pose_1 = SE3(
        xyz=np.array([[2.0], [4.0], [8.0]]),
        roll_pitch_yaw=(np.pi / 2, np.pi / 4, np.pi / 8),
    )

    logger.info(f"Pose 1: {pose_0}")
    logger.info(f"Pose 2: {pose_1}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pose_0.plot(ax)
    pose_1.plot(ax)

    for t in np.arange(0.0, 1.01, 0.1):
        pose_interp = interpolate_se3(pose_1, pose_0, t=t)
        pose_interp.plot(ax)
        logger.info(f"Interpolated Pose at t={t:.2f}: {pose_interp}")

    plt.axis("equal")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

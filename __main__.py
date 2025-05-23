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
    logger.info(pose_0)

    pose_1 = SE3(
        xyz=np.array([[0.0], [1.0], [2.0]]),
        yaw_pitch_roll=(np.pi / 2, np.pi / 4, np.pi / 8),
    )

    pose_interp = interpolate_se3(pose_0, pose_1, t=0.5)

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pose_0.plot(ax)
    pose_1.plot(ax)
    pose_interp.plot(ax)
    plt.axis("equal")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

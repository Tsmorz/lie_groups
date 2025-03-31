"""Basic docstring for my module."""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from se3_group.se3 import SE3, interpolate_se3


def main() -> None:
    """Run a simple demonstration."""
    se3_1 = SE3(
        xyz=np.array([[2.0], [4.0], [8.0]]),
        roll_pitch_yaw=np.array([np.pi / 2, np.pi / 4, np.pi / 8]),
    )
    se3_2 = SE3()

    logger.info(f"SE3 1: {se3_1}")
    logger.info(f"SE3 2: {se3_2}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    se3_1.plot(ax)
    se3_2.plot(ax)

    for t in np.arange(0.0, 1.01, 0.1):
        se3_interp = interpolate_se3(se3_1, se3_2, t=t)
        se3_interp.plot(ax)
        logger.info(f"Interpolated SE3 at t={t:.2f}: {se3_interp}")

    plt.axis("equal")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    plt.show()


if __name__ == "__main__":
    main()

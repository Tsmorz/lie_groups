# se3_group
Functions for the Lie Algebra group SE(3)

$$ SE(3) =
\begin{bmatrix}
    R_{11} & R_{12} & R_{13} & t_1 \\
    R_{21} & R_{22} & R_{23} & t_2 \\
    R_{31} & R_{32} & R_{33} & t_3 \\
    0 & 0 & 0 & 1
\end{bmatrix} $$

where:
- \( $\mathbf{R} \in SO(3) $\) is the **rotation matrix**.
- \( $\mathbf{t} \in \mathbb{R}^3 $\) is the **translation vector**.

This representation is commonly used in **robotics and computer vision** for rigid body transformations.

<p align="center" width="80%">
    <img width="500" alt="Screenshot 2025-03-31 at 9 37 41 AM" src="https://github.com/user-attachments/assets/15d5f56c-32f8-4e18-bb05-bf84c1b48fdc" />
</p>


## Install
To install the library run: `pip install se3_group`

## Development
0. Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
1. `make init` to create the virtual environment and install dependencies
2. `make format` to format the code and check for errors
3. `make test` to run the test suite
4. `make clean` to delete the temporary files and directories
5. `poetry publish --build` to build and publish to https://pypi.org/project/se3_group


## Usage
```
se3_1 = SE3(
    xyz=np.array([[2.0], [4.0], [8.0]]),
    roll_pitch_yaw=np.array([np.pi / 2, np.pi / 4, np.pi / 8]),
)
se3_2 = SE3()

# The classes have a print method
logger.info(f"SE3 1: {se3_1}")
logger.info(f"SE3 2: {se3_2}")

# The poses can be visualized in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
se3_1.plot(ax)
se3_2.plot(ax)

# And you can interpolate the poses
for t in np.arange(0.0, 1.01, 0.1):
    se3_interp = interpolate_se3(se3_1, se3_2, t=t)
    se3_interp.plot(ax)
    logger.info(f"Interpolated SE3 at t={t:.2f}: {se3_interp}")

plt.axis("equal")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.show()

```

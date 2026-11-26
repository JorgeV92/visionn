# vision expansion 

## Filtering, Image Gradients, and Edges

A Sobel filter is a fixed convolution kernel. A CNN learns kernels from data.

An image is a discrete function $I[y, x]$. A $2D$ convolution applies a local kernel $K$. 

$$
(I \ast K)\[y, x\] = \sum_{i,j} I\[y - i, x - j\] \\; K\[i,j\]  
$$

Most deep learning libraries implement cross-correlation instead (no kernel flip) but learnable kernels make the distinction harmless in CNNs/ 

For image gradients: 

$$
  \nabla I =  
  \begin{bmatrix}
  I_x\\
  I_y
  \end{bmatrix},
  \qquad
  \|\nabla I\|_2 = \sqrt{I_x^2 + I_y^2},
  \qquad
  \theta = \mathop{\mathrm{atan2}}(I_y, I_x)
$$

Sobel kernels approximate horizontal and vertical derivatives while smoothing in the perpendicular direction:

$$
K_x =
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\quad
K_y =
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}
$$

Key idea: differentiation amplifies noise, so smooth first with a Gaussian \(G_\sigma\), then differentiate:

$$
\nabla(G_\sigma * I)
$$

This is the core intuition behind Canny edges and remains relevant in stereo matching, visual odometry, lane perception, and image-quality debugging for learned systems.


### Paper and technical resource (25 min)

- **Paper — peer-reviewed:** John Canny, “[A Computational Approach to Edge Detection](https://ieeexplore.ieee.org/document/4767851)” (IEEE TPAMI, 1986). Read the abstract, introduction, and the sections explaining detection, localization, and single-response criteria. Canny formalizes the tradeoff between reliably detecting a noisy edge and placing it precisely. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/4767851?signout=success\&utm_source=chatgpt.com)

- **Technical resource:** Stanford CS231n, “[Convolutional Networks](https://cs231n.github.io/convolutional-networks/).” Focus on local receptive fields, parameter sharing, stride, padding, and output shapes. It makes the bridge from hand-designed filters to learned convolutional features explicit. [cs231n.github.io](https://cs231n.github.io/convolutional-networks/?utm_source=chatgpt.com)

Before reading: be comfortable with matrix multiplication, partial derivatives, and the idea that a kernel is a local weighted sum.

```python
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)

SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]]
)

def gaussian_kernel_1d(sigma: float, radius: int) -> torch.Tensor:
    """Return a normalized 1-D Gaussian kernel of length 2*radius + 1."""
    # TODO
    raise NotImplementedError

def reflect_conv2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Cross-correlate one grayscale [H, W] image with one [kH, kW] kernel.
    Preserve H, W using reflect padding.
    """
    # TODO: validate dimensions, add batch/channel dimensions,
    # reflect-pad, apply F.conv2d, then remove batch/channel dimensions.
    raise NotImplementedError

def sobel_edges(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gx, gy, and gradient magnitude for a grayscale image."""
    # TODO
    raise NotImplementedError

def threshold_edges(magnitude: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return a boolean edge map."""
    # TODO
    raise NotImplementedError
```

Create `test_edges.py`:

```python
import torch
from edges import gaussian_kernel_1d, reflect_conv2d, sobel_edges, threshold_edges

def test_gaussian_is_symmetric_and_normalized():
    k = gaussian_kernel_1d(sigma=1.0, radius=3)
    assert k.shape == (7,)
    assert torch.allclose(k, torch.flip(k, dims=[0]))
    assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-6)

def test_constant_image_has_no_interior_gradient():
    image = torch.ones(9, 9)
    _, _, mag = sobel_edges(image)
    assert torch.allclose(mag[1:-1, 1:-1], torch.zeros(7, 7), atol=1e-6)

def test_vertical_step_has_vertical_edge_response():
    image = torch.zeros(11, 11)
    image[:, 5:] = 1.0
    gx, gy, mag = sobel_edges(image)

    assert mag[:, 4:6].mean() > 1.0
    assert gy[2:-2, 2:4].abs().mean() < 1e-5

def test_threshold_returns_bool_map():
    mag = torch.tensor([[0.2, 0.8], [1.1, 0.0]])
    edges = threshold_edges(mag, threshold=0.8)
    assert edges.dtype == torch.bool
    assert edges.tolist() == [[False, False], [True, False]]
```

Run:

```bash
python -m pytest -q
```

Expected behavior:

- A constant image has essentially zero interior gradient.
- A vertical brightness step creates a strong $I_x$ response near the boundary.
- Increasing the threshold produces fewer edges.

Debugging hints:

- `F.conv2d` expects `[N, C, H, W]` input and `[out_channels, in_channels, kH, kW]` weights.
- Check every tensor’s shape before calling `conv2d`.
- Do not normalize Sobel kernels; their scale is meaningful for these tests.
- If the vertical-step test fails, verify which spatial axis your kernel differentiates.

Optional extension: add Gaussian smoothing before Sobel, inject Gaussian noise into the step image, and measure how precision/recall changes as you vary \(\sigma\) and the threshold.

### Graduate-level questions

1. Why does differentiation amplify high-frequency noise? Explain in both spatial and Fourier-domain terms.

2. A camera observes a sharp edge blurred by motion and sensor noise. Why might a larger Gaussian scale improve detection but worsen localization?

3. A CNN learns a first-layer edge-like kernel. Give two reasons it may outperform Sobel, and two reasons a fixed Sobel baseline is still useful in a perception pipeline.

4. Design a small experiment comparing Sobel-with-thresholding against Canny on lane-boundary images. Specify metrics, likely failure cases, and how you would avoid tuning on the test set.

### Review and progress tracking 

Takeaways:

- Convolution is local feature extraction; CNNs generalize it through learned filters.
- Gradients encode edge strength and orientation.
- Smoothing versus localization is a fundamental perception tradeoff.
- Classical edge detectors remain valuable baselines and diagnostic tools.

Further reading: Szeliski’s free textbook, [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/), Chapters 3–4.

## Pinhole Cameras, Projection, and Calibration 

This builds on image gradients: after extracting image structure, geometry tells us how a 3D scene produced those pixels.

### Concepts and mathematics 

A 3D point in camera coordinates $X_c = (X,Y,Z)^T$ projects to normalized image coordinates:

$$
x = \frac{X}{Z}, \qquad y = \frac{Y}{Z}
$$

The camera intrinsic matrix maps normalized coordinates to pixels:

$$
K =
\begin{bmatrix}
f_x & s & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

With camera pose $R,t$, the complete projective camera is:

$$
\tilde p \sim K[R \mid t]\tilde X = P\tilde X
$$

where $\tilde X=(X,Y,Z,1)^T$ and $\tilde p=(u,v,w)^T$. Convert back to pixels using:

$$
u = \frac{\tilde p_1}{\tilde p_3}, \qquad
v = \frac{\tilde p_2}{\tilde p_3}
$$

Important intuition:

- $K$: camera’s internal pixel geometry.
- $R,t$: where the camera is and where it points.
- Perspective division causes distant objects to appear smaller.
- A single image loses absolute depth: many 3D points along the same camera ray yield the same pixel.

For autonomous driving, calibration errors create systematic projection errors between cameras, LiDAR, HD maps, and detected objects.

### Paper and technical resource 

- **Paper — peer-reviewed:** Zhengyou Zhang, “[A Flexible New Technique for Camera Calibration](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/)” (IEEE TPAMI, 2000). It estimates camera intrinsics from several images of a planar calibration pattern, then refines them with nonlinear optimization and models radial distortion. [microsoft.com](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/?utm_source=chatgpt.com)

- **Technical resource:** Stanford CS231A, “[Camera Models](https://web.stanford.edu/class/archive/cs/cs231a/cs231a.1246/course_notes/01-camera-models.pdf).” Read Sections 2–4 on pinhole cameras, homogeneous coordinates, intrinsics, and extrinsics. [web.stanford.edu](https://web.stanford.edu/class/archive/cs/cs231a/cs231a.1246/course_notes/01-camera-models.pdf?utm_source=chatgpt.com)

Prerequisites: matrix multiplication, linear systems, SVD at a high level, and homogeneous coordinates.

### Practical implementation

Implement projection and a Direct Linear Transform (DLT) camera estimator in PyTorch. This uses synthetic points, so it runs comfortably on your Mac.

Create `camera.py`:

```python
import torch

def homogenize(points: torch.Tensor) -> torch.Tensor:
    """Convert [N, D] Euclidean points to [N, D+1] homogeneous points."""
    # TODO
    raise NotImplementedError

def dehomogenize(points: torch.Tensor) -> torch.Tensor:
    """Convert [N, D+1] homogeneous points to [N, D]."""
    # TODO: divide every coordinate except the final one by final coordinate.
    raise NotImplementedError

def project(P: torch.Tensor, points_3d: torch.Tensor) -> torch.Tensor:
    """
    Project [N, 3] world points using a [3, 4] camera matrix.
    Return pixel coordinates [N, 2].
    """
    # TODO
    raise NotImplementedError

def estimate_camera_dlt(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate P [3, 4] from N >= 6 non-coplanar 3D-to-2D correspondences.

    Build the homogeneous linear system A p = 0, solve with SVD,
    reshape its last right-singular vector into P, and normalize scale.
    """
    # TODO
    raise NotImplementedError
```

Create `test_camera.py`:

```python
import torch
from camera import homogenize, dehomogenize, project, estimate_camera_dlt

P_TRUE = torch.tensor([
    [800.0,   0.0, 320.0,  40.0],
    [  0.0, 810.0, 240.0, -25.0],
    [  0.0,   0.0,   1.0,   0.2],
], dtype=torch.float64)

POINTS_3D = torch.tensor([
    [-1.0, -1.0, 4.0], [ 1.0, -1.0, 4.5],
    [-1.0,  1.0, 5.0], [ 1.0,  1.0, 5.5],
    [ 0.3, -0.7, 6.0], [-0.8,  0.4, 6.5],
    [ 0.7,  0.2, 7.0], [-0.2, -0.4, 7.5],
], dtype=torch.float64)

def test_homogeneous_round_trip():
    x = torch.tensor([[2.0, -3.0], [0.5, 4.0]])
    assert torch.allclose(dehomogenize(homogenize(x)), x)

def test_projection_returns_pixels():
    pixels = project(P_TRUE, POINTS_3D)
    assert pixels.shape == (8, 2)
    assert torch.isfinite(pixels).all()

def test_dlt_reprojects_training_points():
    pixels = project(P_TRUE, POINTS_3D)
    P_est = estimate_camera_dlt(POINTS_3D, pixels)
    reprojection = project(P_est, POINTS_3D)

    assert P_est.shape == (3, 4)
    assert torch.allclose(reprojection, pixels, atol=1e-6)

def test_dlt_generalizes_to_held_out_point():
    pixels = project(P_TRUE, POINTS_3D)
    P_est = estimate_camera_dlt(POINTS_3D, pixels)

    held_out = torch.tensor([[0.25, -0.10, 6.25]], dtype=torch.float64)
    assert torch.allclose(
        project(P_est, held_out),
        project(P_TRUE, held_out),
        atol=1e-5,
    )
```

Run:

```bash
python -m pytest -q
```

Debugging hints:

- Keep all tensors `float64`; DLT can be numerically sensitive.
- \(P\) is only defined up to nonzero scale. Test reprojections, not raw matrix equality.
- Each 3D–2D correspondence contributes two rows to \(A\).
- Do not use coplanar points for this version of DLT.

Optional extension: add radial distortion,

$$
x_d = x(1 + k_1r^2 + k_2r^4), \quad
y_d = y(1 + k_1r^2 + k_2r^4)
$$

then measure how an unmodeled distortion changes reprojection error.

### Graduate-level questions 

1. Why can’t a single calibrated image determine metric depth without additional assumptions or measurements?

2. Derive why scaling \(P\) by any nonzero constant does not change projected pixel coordinates.

3. A truck has a forward camera and roof LiDAR. What are the separate roles of camera intrinsics and camera-to-LiDAR extrinsics? Which errors cause a pixel-level misalignment that grows with distance?

4. Why is estimating a camera matrix from noisy correspondences an optimization problem rather than an exact linear-algebra problem in practice?

###  Review and progress tracking 

Key takeaways:

- Projection is linear in homogeneous coordinates, then nonlinear after perspective division.
- \(P=K[R\mid t]\) separates internal camera geometry from pose.
- Calibration is required before reliable metric reconstruction or sensor fusion.
- DLT is a useful initialization, while real calibration needs normalization and nonlinear refinement.

Further reading: Stanford’s [camera-calibration lecture](https://web.stanford.edu/class/cs231a/lectures_2025/lecture3_camera_calibration.pdf).




## Epipolar Geometry and the Fundamental Matrix 

This extends camera calibration to two views: calibration tells you how a camera projects; epipolar geometry constrains where a point from one camera may appear in another.

### Part 1 — Concepts and mathematics (25 min)

For a corresponding pair of homogeneous image points $\tilde x_1,\tilde x_2$,

$$
\tilde x_2^T F \tilde x_1 = 0
$$

where $F\in\mathbb{R}^{3\times3}$ is the **fundamental matrix**. Given a point in image 1, its match in image 2 must lie on the epipolar line:

$$
\ell_2 = F\tilde x_1
$$

If both cameras are calibrated, use the essential matrix:

$$
E = [t]_\times R,
\qquad
F = K_2^{-T} E K_1^{-1}
$$

where $[t]_\times$ is the skew-symmetric matrix representing cross products with translation $t$.

Important facts:

- $F$ has rank 2.
- Epipolar geometry reduces 2D correspondence search to a 1D line search.
- Pure rotation has degenerate translation geometry: no usable triangulation baseline.
- This is central to stereo depth, visual odometry, SLAM initialization, and multi-camera calibration.

### Paper and technical resource

- **Paper — peer-reviewed:** Richard Hartley, “[In Defense of the Eight-Point Algorithm](https://users.cecs.anu.edu.au/~hartley/My-Papers.html)” (IEEE TPAMI, 1997). Main lesson: point normalization before solving the linear system dramatically improves conditioning and makes the eight-point method practical. [cse.unr.edu](https://www.cse.unr.edu/~bebis/CS485/Handouts/hartley.pdf?utm_source=chatgpt.com)

- **Technical resource:** Stanford CS231A, “[Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf).” Read through the derivation of the epipolar constraint and the distinction between \(E\) and \(F\). [web.stanford.edu](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf?utm_source=chatgpt.com)

Before reading: review homogeneous coordinates, SVD, camera matrices, and the DLT idea from the previous session.

### Practical implementation 

Implement the **normalized eight-point algorithm** in PyTorch. This is a compact version of a real geometry-estimation component.

Create `fundamental.py`:

```python
import torch

def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """Convert [N, 2] points into [N, 3] homogeneous points."""
    # TODO
    raise NotImplementedError

def normalize_points(points: torch.Tensor):
    """
    Normalize image points so their centroid is zero and their mean distance
    from the origin is sqrt(2).

    Return normalized points [N, 2] and transform T [3, 3], where:
        x_normalized_h = T @ x_h
    """
    # TODO
    raise NotImplementedError

def estimate_fundamental_8point(
    points1: torch.Tensor,
    points2: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate F such that x2.T @ F @ x1 = 0.

    Requirements:
    - N >= 8 corresponding [N, 2] points
    - normalize both point sets before building A
    - solve A f = 0 using SVD
    - enforce rank(F) = 2
    - denormalize and normalize F by its Frobenius norm
    """
    # TODO
    raise NotImplementedError

def sampson_error(
    F: torch.Tensor,
    points1: torch.Tensor,
    points2: torch.Tensor,
) -> torch.Tensor:
    """Return one first-order geometric error per correspondence."""
    # TODO
    raise NotImplementedError
```

Create `test_fundamental.py`:

```python
import torch
from fundamental import (
    to_homogeneous,
    normalize_points,
    estimate_fundamental_8point,
    sampson_error,
)

def test_normalization_properties():
    points = torch.tensor(
        [[10.0, 20.0], [50.0, 30.0], [20.0, 80.0], [90.0, 15.0]],
        dtype=torch.float64,
    )
    normalized, _ = normalize_points(points)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2, dtype=torch.float64))
    mean_distance = torch.linalg.vector_norm(normalized, dim=1).mean()
    assert torch.allclose(mean_distance, torch.tensor(2.0**0.5), atol=1e-10)

def test_estimated_fundamental_matrix_has_rank_two():
    torch.manual_seed(7)
    points1 = torch.rand(12, 2, dtype=torch.float64) * 640.0
    points2 = points1 + torch.tensor([[-15.0, 0.0]], dtype=torch.float64)

    F = estimate_fundamental_8point(points1, points2)
    singular_values = torch.linalg.svdvals(F)

    assert F.shape == (3, 3)
    assert singular_values[-1] < 1e-9

def test_epipolar_constraint_on_synthetic_rectified_pair():
    torch.manual_seed(4)
    points1 = torch.rand(16, 2, dtype=torch.float64) * 500.0
    points2 = points1 + torch.tensor([[-20.0, 0.0]], dtype=torch.float64)

    F = estimate_fundamental_8point(points1, points2)
    x1 = to_homogeneous(points1)
    x2 = to_homogeneous(points2)

    residuals = torch.einsum("ni,ij,nj->n", x2, F, x1).abs()
    assert residuals.max() < 1e-7

def test_sampson_error_is_nonnegative():
    points1 = torch.tensor([[100.0, 50.0], [120.0, 80.0]], dtype=torch.float64)
    points2 = torch.tensor([[90.0, 50.0], [110.0, 80.0]], dtype=torch.float64)
    F = estimate_fundamental_8point(
        torch.cat([points1] * 4),
        torch.cat([points2] * 4),
    )
    assert torch.all(sampson_error(F, points1, points2) >= 0)
```

Run:

```bash
python -m pytest -q
```

Debugging hints:

- Use `float64`.
- Be consistent with the convention $x_2^T F x_1=0$.
- The last right-singular vector of $A$ is the solution for flattened $F$.
- Enforce rank 2 by setting the smallest singular value of $F$ to zero.
- The final test deliberately has repeated points; if it causes instability, explain why this is a bad estimator input and replace it with distinct correspondences for your own validation.

Optional extension: add Gaussian pixel noise and compare the raw versus normalized eight-point algorithms by mean Sampson error.

### Graduate-level questions 

1. Why does a point in one image correspond to an epipolar line—not usually a point—in the second image?

2. Explain why $F$ has rank 2. What does its null space represent geometrically?

3. A stereo rig’s baseline doubles while its focal length and measurement noise remain fixed. How should expected depth accuracy change for distant points?

4. For autonomous driving, why would RANSAC be paired with fundamental-matrix estimation? Name two likely outlier sources.

### Review and progress tracking

Key takeaways:

- Two-view geometry converts correspondence from a 2D search to a 1D epipolar search.
- $F$ applies to uncalibrated cameras; $E$ uses calibrated camera coordinates.
- Point normalization is an algorithmic necessity, not merely a cosmetic preprocessing step.
- Geometry estimation needs robust outlier handling in real scenes.

Further reading: Stanford’s [epipolar-geometry lecture](https://web.stanford.edu/class/cs231a/lectures_2025/lecture5_epipolar_geometry.pdf).

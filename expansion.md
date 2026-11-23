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

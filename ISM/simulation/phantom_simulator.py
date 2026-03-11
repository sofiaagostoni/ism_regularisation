import torch
import numpy as np
from torch.fft import irfftn, rfftn, ifftshift
from torch import real, einsum

from scipy.ndimage import gaussian_filter


def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk',  dim3: str = None, fourier: tuple = (False, False),
                             padding: list = None):
    kernel = torch.as_tensor(kernel)
    volume = torch.as_tensor(volume)
    if dim3 is None:
        dim3 = dim1 + dim2
        dim3 = ''.join(sorted(set(dim3), key=dim3.index))
    elif not isinstance(dim3, str):
        raise ValueError("dim3 must be a string")
    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]
    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]
    if fourier[0] == False:
        #kernel_shifted = ifftshift(kernel, dim=axis_list[0])  # Centra il kernel
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel
    if fourier[1] == False:
        #volume_shifted = ifftshift(volume, dim=axis_list[1])  # Centra il volume
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume
    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)
    conv = irfftn(conv, dim=axis_list[2], s=padding)  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part
    return conv
def disk(radius, shape):
    nx, ny = shape
    x = np.arange(-nx // 2, nx // 2, 1)
    y = np.arange(-nx // 2, nx // 2, 1)

    xx, yy = np.meshgrid(x, y)

    r = np.sqrt(xx ** 2 + yy ** 2)

    return np.where(r <= radius, 1, 0)


def sparse_random_image(shape, num_nonzero, value_range=(0.1, 1.0), margin=0, dtype=np.float32):
    """
    Generate an array of zeros with `num_nonzero` random pixels set to
    values uniformly drawn from `value_range`. Random pixels are restricted
    to lie at least `margin` pixels away from each border.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the array.
    num_nonzero : int
        Number of nonzero pixels.
    value_range : (float, float)
        Range [low, high] for random values.
    margin : int
        Excluded border width (applied to all dimensions).
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    img : np.ndarray
        Generated sparse image.
    """
    shape = tuple(shape)
    ndim = len(shape)

    # Valid ranges for each dimension
    valid_ranges = [range(margin, s - margin) for s in shape]
    if any(len(r) <= 0 for r in valid_ranges):
        raise ValueError("Margin too large for given shape.")

    # Total number of valid positions
    valid_coords = np.array(np.meshgrid(*valid_ranges, indexing="ij")).reshape(ndim, -1).T
    if num_nonzero > len(valid_coords):
        raise ValueError(f"num_nonzero={num_nonzero} exceeds available {len(valid_coords)} valid positions.")

    # Choose random coordinates
    chosen_idx = np.random.choice(len(valid_coords), size=num_nonzero, replace=False)
    chosen_coords = valid_coords[chosen_idx]

    # Fill image
    img = np.zeros(shape, dtype=dtype)
    values = np.random.uniform(value_range[0], value_range[1], num_nonzero).astype(dtype)
    img[tuple(chosen_coords.T)] = values

    return img


def generate_disk_phantom(shape, num, radius, value_range=(0.1, 1.0), smooth: bool = False):

    img = sparse_random_image(shape=shape, num_nonzero=num, value_range=value_range, margin = radius)
    kernel = disk(radius=radius, shape=shape)

    phantom = partial_convolution_rfft(kernel, img, 'xy', 'xy', 'xy', fourier=(0,0))

    if smooth is True:
        phantom = gaussian_filter(phantom, sigma=1)

    phantom[phantom < 0] = 0

    return phantom

def IIT_G2DFit_gaussian2D(x, y, sx, sy, ux, uy, K):
    result = K / (2.0 * np.pi * sx * sy) * np.exp(-((x - ux)**2 / (2.0 * sx**2)) - ((y - uy)**2 / (2.0 * sy**2)))
    return result

def IIT_G2DFit_gaussian2DDraw(w, h, sx, sy, ux, uy, theta, K, pixelSize):
    result = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            xp = x * pixelSize - ux
            yp = y * pixelSize - uy
            xpp = xp * np.cos(theta) - yp * np.sin(theta)
            ypp = xp * np.sin(theta) + yp * np.cos(theta)
            result[y, x] = IIT_G2DFit_gaussian2D(xpp, ypp, sx, sy, 0, 0, K)
    return result

def Nucleus(shape, noOfGauss, value_range, std_range, ellipse_axes, background_noise=0.01, seed=None):
    
    # Parameters
    dimImx = shape[0] // 2
    dimImy = shape[1] // 2
    minAmpGauss = value_range[0]
    maxAmpGauss = value_range[1]
    stdMin = std_range[0]
    stdMax = std_range[1]
    
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    y, x = np.indices(shape)
    cy, cx = h / 2, w / 2  # center of image

    # --- Define elliptical mask for the nucleus ---
    ellipse_mask = ((x - cx) / ellipse_axes[0])**2 + ((y - cy) / ellipse_axes[1])**2 <= 1

    # --- Initialize empty image ---
    phantom = np.zeros(shape, dtype=float)

    # --- Random positions for blobs inside the ellipse ---
    positions = []
    while len(positions) < noOfGauss:
        px = np.random.uniform(0, w)
        py = np.random.uniform(0, h)
        if ellipse_mask[int(py % h), int(px % w)]:
            positions.append((px, py))
    positions = np.array(positions)

    # --- Add Gaussian blobs efficiently ---
    for (px, py) in positions:
        amp = np.random.uniform(*value_range)
        std = np.random.uniform(*std_range)
        # Create Gaussian around position (vectorized local window)
        xg = np.exp(-((x - px)**2 + (y - py)**2) / (2 * std**2))
        phantom += amp * xg

    # --- Mask everything outside ellipse ---
    phantom *= ellipse_mask

    # --- Add smoothness + background noise for realism ---
    phantom = gaussian_filter(phantom, sigma=1.0)
    phantom += background_noise * np.random.random(shape)

    # --- Normalize to [0, 1] for display ---
    phantom -= phantom.min()
    phantom /= phantom.max()

    return phantom


def generate_membrane_phantom(shape, radius, thickness, value_range, noise_level=0.05, blur_sigma=1.0, seed=None):
    """
    Generate a synthetic 2D fluorescence-like image of a cell membrane
    as a bright elliptical shell with slight intensity variation and blur.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = shape
    y, x = np.indices(shape)
    cy, cx = h / 2, w / 2

    # Ellipse parameters
    r1, r2 = radius
    a = r1  # x-axis radius
    b = r2  # y-axis radius

    # Distance from ellipse boundary (approximation)
    ellipse_eq = ((x - cx) / a)**2 + ((y - cy) / b)**2
    membrane_mask = np.exp(-((ellipse_eq - 1.0)**2) / (2 * (thickness / 100)**2))

    # Random intensity variations along the membrane
    rand_texture = np.random.uniform(*value_range, size=shape)
    phantom = membrane_mask * rand_texture

    # Smooth slightly to look more optical
    phantom = gaussian_filter(phantom, sigma=blur_sigma)

    # Add background noise
    phantom += noise_level * np.random.random(shape)

    # Normalize
    phantom -= phantom.min()
    phantom /= phantom.max()

    return phantom

def generate_hollow_membrane_phantom(
    shape=(256, 256),
    n_ellipses=40,
    radius_range=(6, 25),
    amplitude_range=(0.5, 1.0),
    thickness=0.15,
    blur_sigma=1.0,
    noise_level=0.05,
    seed=None
):
    """
    Generate a 2D fluorescence-like phantom with multiple small hollow elliptical
    membranes (e.g., simulating mitochondria or vesicles).
    Each ellipse is randomly placed, oriented, and sized, with soft blurred edges.
    """
    if seed is not None:
        np.random.seed(seed)
    h, w = shape
    y, x = np.indices(shape)
    phantom = np.zeros(shape, dtype=float)
    for _ in range(n_ellipses):
        # Random ellipse parameters
        cx = np.random.uniform(0.2 * w, 0.8 * w)
        cy = np.random.uniform(0.2 * h, 0.8 * h)
        rx = np.random.uniform(*radius_range)
        ry = np.random.uniform(*radius_range)
        amp = np.random.uniform(*amplitude_range)
        theta = np.random.uniform(0, np.pi)
        # Rotate coordinate system
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X = x - cx
        Y = y - cy
        Xr = cos_t * X + sin_t * Y
        Yr = -sin_t * X + cos_t * Y
        # Elliptical distance field
        ellipse_eq = (Xr / rx)**2 + (Yr / ry)**2
        # Hollow ring profile (soft Gaussian shell)
        ring = np.exp(-((ellipse_eq - 1.0)**2) / (2 * thickness**2))
        # Random local modulation to simulate uneven labeling
        ring *= amp * np.random.uniform(0.8, 1.2, size=shape)
        phantom += ring
    # Add Gaussian blur for optical softness
    phantom = gaussian_filter(phantom, sigma=blur_sigma)
    # Add background noise
    phantom += noise_level * np.random.random(shape)
    # Normalize to [0,1]
    phantom -= phantom.min()
    phantom /= phantom.max()
    return phantom
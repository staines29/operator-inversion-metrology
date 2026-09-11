"""
src/forward.py
Forward modeling module for 1D sub-wavelength semiconductor grating metrology.
Simulates a discrete optical forward blur operator (Toeplitz convolution)
and corrupted sensor observations under additive Gaussian noise.
"""

import numpy as np
from scipy.linalg import toeplitz

def generate_grating_profile(n_points: int = 128, pitch_bins: int = 32, duty_cycle: float = 0.5) -> np.ndarray:
    """Generates a 1D binary nanoscale surface-relief grating profile x in {0, 1}^N."""
    x = np.zeros(n_points, dtype=np.float64)
    line_width = int(pitch_bins * duty_cycle)
    
    for start in range(0, n_points, pitch_bins):
        end = min(start + line_width, n_points)
        x[start:end] = 1.0
        
    return x

def build_forward_toeplitz_operator(n_points: int = 128, kernel_sigma: float = 3.5) -> np.ndarray:
    """Constructs a 1D discrete optical blur operator matrix A (Toeplitz convolution)."""
    radius = n_points // 2
    grid = np.arange(-radius, radius, dtype=np.float64)
    psf_kernel = np.exp(-0.5 * (grid / kernel_sigma) ** 2)
    psf_kernel /= np.sum(psf_kernel)

    first_col = np.zeros(n_points, dtype=np.float64)
    half_width = len(psf_kernel) // 2
    
    for i in range(len(psf_kernel)):
        idx = (i - half_width) % n_points
        first_col[idx] = psf_kernel[i]
        
    A = toeplitz(first_col, first_col)
    return A

def simulate_measurement(A: np.ndarray, x: np.ndarray, noise_sigma: float = 0.02, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Executes the forward optical acquisition: y = A * x + eta."""
    np.random.seed(seed)
    y_clean = A @ x
    noise = np.random.normal(0.0, noise_sigma, size=y_clean.shape)
    y_noisy = y_clean + noise
    return y_noisy, y_clean
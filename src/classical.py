"""
src/classical.py
Classical linear inverse solvers for the ill-conditioned operator A.
"""

import numpy as np

def compute_operator_diagnostics(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Computes SVD singular values and condition number."""
    U, s, Vt = np.linalg.svd(A)
    condition_number = float(s[0] / s[-1]) if s[-1] != 0.0 else float("inf")
    return s, condition_number

def naive_inversion(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes unregularized pseudo-inverse solution."""
    return np.linalg.pinv(A) @ y

def tikhonov_l2_inversion(A: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    """Solves standard 0th-order Tikhonov regularized objective (Ridge regression)."""
    n = A.shape[1]
    AtA = A.T @ A
    regularizer = alpha * np.eye(n)
    return np.linalg.solve(AtA + regularizer, A.T @ y)

def tikhonov_first_order_inversion(A: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    """Solves 1st-order Tikhonov regularized objective penalizing discrete gradients."""
    n = A.shape[1]
    D = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
        
    AtA = A.T @ A
    DtD = D.T @ D
    return np.linalg.solve(AtA + alpha * DtD, A.T @ y)
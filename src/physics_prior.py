"""
src/physics_prior.py
Physics-informed gradient-descent solver using PyTorch autograd.
Minimizes data discrepancy with an isotropic Total Variation (TV) prior.
"""

import torch
import numpy as np

def solve_total_variation(
    A: np.ndarray,
    y: np.ndarray,
    lambda_tv: float = 5e-3,
    lr: float = 0.05,
    max_iters: int = 1500
) -> np.ndarray:
    A_tensor = torch.from_numpy(A).float()
    y_tensor = torch.from_numpy(y).float()
    
    x_hat = torch.zeros(A.shape[1], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x_hat], lr=lr)
    eps = 1e-8
    
    for _ in range(max_iters):
        optimizer.zero_grad()
        
        residual = torch.matmul(A_tensor, x_hat) - y_tensor
        data_fidelity = 0.5 * torch.mean(residual ** 2)
        
        diff = x_hat[1:] - x_hat[:-1]
        tv_penalty = torch.sum(torch.sqrt(diff ** 2 + eps))
        
        loss = data_fidelity + lambda_tv * tv_penalty
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            x_hat.clamp_(0.0, 1.0)
            
    return x_hat.detach().cpu().numpy().astype(np.float64)
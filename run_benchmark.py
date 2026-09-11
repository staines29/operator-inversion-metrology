"""
run_benchmark.py
Executes the comparative inversion suite and reports benchmark metrics.
Publication-grade visual configuration: zero grid clutter, flush origin (x=0).
"""

import numpy as np
import matplotlib.pyplot as plt
from src.forward import generate_grating_profile, build_forward_toeplitz_operator, simulate_measurement
from src.classical import compute_operator_diagnostics, naive_inversion, tikhonov_l2_inversion, tikhonov_first_order_inversion
from src.physics_prior import solve_total_variation

def relative_l2_error(x_true: np.ndarray, x_est: np.ndarray) -> float:
    return float(np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true))

def main():
    print("=" * 58)
    print("ARCNL METROLOGY SANDBOX: DISCRETE OPERATOR INVERSION")
    print("=" * 58)
    
    N = 128
    x_true = generate_grating_profile(n_points=N, pitch_bins=32, duty_cycle=0.5)
    A = build_forward_toeplitz_operator(n_points=N, kernel_sigma=3.5)
    y_noisy, y_clean = simulate_measurement(A, x_true, noise_sigma=0.03, seed=101)
    
    singular_values, cond_num = compute_operator_diagnostics(A)
    print(f"[*] Matrix Dimensions     : {A.shape[0]} x {A.shape[1]}")
    print(f"[*] Condition Number k(A) : {cond_num:.2e}")
    print(f"[*] Max Singular Value s_0: {singular_values[0]:.4f}")
    print(f"[*] Min Singular Value s_N: {singular_values[-1]:.4e}")
    print("-" * 58)
    
    print("[+] Computing Naive Pseudo-Inverse...")
    x_naive = naive_inversion(A, y_noisy)
    
    print("[+] Computing Tikhonov L2 Regularization (alpha=1e-2)...")
    x_tik_l2 = tikhonov_l2_inversion(A, y_noisy, alpha=1e-2)
    
    print("[+] Computing Tikhonov 1st-Order Regularization (alpha=1e-2)...")
    x_tik_d1 = tikhonov_first_order_inversion(A, y_noisy, alpha=1e-2)
    
    print("[+] Computing Physics-Informed Total Variation (PyTorch)...")
    x_tv = solve_total_variation(A, y_noisy, lambda_tv=1e-3, lr=0.05, max_iters=2000)
    print("-" * 58)
    
    err_naive = relative_l2_error(x_true, x_naive)
    err_tik_l2 = relative_l2_error(x_true, x_tik_l2)
    err_tik_d1 = relative_l2_error(x_true, x_tik_d1)
    err_tv = relative_l2_error(x_true, x_tv)
    
    print(f"[-] Relative L2 Error (Naive Inversion) : {err_naive:.4e}")
    print(f"[-] Relative L2 Error (Tikhonov L2)     : {err_tik_l2:.4f}")
    print(f"[-] Relative L2 Error (Tikhonov 1st-Ord): {err_tik_d1:.4f}")
    print(f"[-] Relative L2 Error (Total Variation) : {err_tv:.4f}")
    print("=" * 58)
    
    # -------------------------------------------------------------
    # Visual Scaffolding: Flush Origin, No Grids, Publication Grade
    # -------------------------------------------------------------
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#2c3e50'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=300)
    
    # Panel 1: Optical Forward Degradation
    ax1 = axes[0, 0]
    ax1.plot(x_true, color='black', linewidth=2.0, label='Ground Truth Grating x')
    ax1.plot(y_noisy, color='gray', linestyle='--', linewidth=1.2, label='Noisy Sensor y (Blurred)')
    ax1.set_title('Optical Acquisition & Forward Degradation', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, N - 1)
    ax1.set_ylim(-0.08, 1.12)
    ax1.grid(False)
    ax1.legend(loc='center right', frameon=True)
    
    # Panel 2: Naive Inversion Exploding Under Noise
    ax2 = axes[0, 1]
    ax2.plot(x_true, color='black', linewidth=1.2, alpha=0.6, label='Ground Truth')
    ax2.plot(x_naive, color='#e74c3c', linewidth=0.8, alpha=0.8, label=f'Naive Inversion (Err: {err_naive:.2e})')
    ax2.set_title('Ill-Posed Divergence (Naive Pseudo-Inverse)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, N - 1)
    ax2.set_ylim(-2.0, 3.0)
    ax2.grid(False)
    ax2.legend(loc='upper right', frameon=True)
    
    # Panel 3: Classical Tikhonov (L2 and 1st-Order Smoothing)
    ax3 = axes[1, 0]
    ax3.plot(x_true, color='black', linewidth=1.2, alpha=0.5, label='Ground Truth')
    ax3.plot(x_tik_l2, color='blue', linewidth=1.5, label=f'Tikhonov L2 (Err: {err_tik_l2:.2f})')
    ax3.plot(x_tik_d1, color='#f39c12', linewidth=1.5, label=f'Tikhonov 1st-Ord (Err: {err_tik_d1:.2f})')
    ax3.set_title('Classical Regularization (Tikhonov)', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, N - 1)
    ax3.set_ylim(-0.25, 1.35)
    ax3.grid(False)
    ax3.legend(loc='upper right', frameon=True)
    
    # Panel 4: Physics-Informed Variational Inversion (Total Variation)
    ax4 = axes[1, 1]
    ax4.plot(x_true, color='black', linewidth=2.0, label='Ground Truth')
    ax4.plot(x_tv, color='#1e824c', linewidth=2.0, label=f'Total Variation Prior (Err: {err_tv:.2f})')
    ax4.set_title('Physics-Informed Edge-Preserving Reconstruction (TV)', fontsize=11, fontweight='bold')
    ax4.set_xlim(0, N - 1)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(False)
    ax4.legend(loc='center right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('metrology_inversion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Publication-grade figure saved: metrology_inversion.png")

if __name__ == "__main__":
    main()
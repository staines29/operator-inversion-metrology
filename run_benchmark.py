"""
run_benchmark.py
Executes the comparative inversion suite and reports benchmark metrics.
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
    
    print(f"[-] Relative L2 Error (Naive Inversion) : {err_naive:.4f}")
    print(f"[-] Relative L2 Error (Tikhonov L2)     : {err_tik_l2:.4f}")
    print(f"[-] Relative L2 Error (Tikhonov 1st-Ord): {err_tik_d1:.4f}")
    print(f"[-] Relative L2 Error (Total Variation) : {err_tv:.4f}")
    print("=" * 58)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    
    axs[0, 0].plot(x_true, label="Ground Truth Grating x", color="black", linewidth=2)
    axs[0, 0].plot(y_noisy, label="Noisy Sensor y (Blurred)", color="gray", linestyle="--", alpha=0.8)
    axs[0, 0].set_title("Optical Acquisition & Forward Degradation")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    axs[0, 1].plot(x_true, label="Ground Truth", color="black", linewidth=1.5, alpha=0.5)
    axs[0, 1].plot(x_naive, label=f"Naive Inversion (Err: {err_naive:.2f})", color="red", alpha=0.7)
    axs[0, 1].set_title("Ill-Posed Divergence (Naive Pseudo-Inverse)")
    axs[0, 1].set_ylim([-2, 3])
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    axs[1, 0].plot(x_true, label="Ground Truth", color="black", linewidth=1.5, alpha=0.5)
    axs[1, 0].plot(x_tik_l2, label=f"Tikhonov L2 (Err: {err_tik_l2:.2f})", color="blue")
    axs[1, 0].plot(x_tik_d1, label=f"Tikhonov 1st-Ord (Err: {err_tik_d1:.2f})", color="orange")
    axs[1, 0].set_title("Classical Regularization (Tikhonov)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    axs[1, 1].plot(x_true, label="Ground Truth", color="black", linewidth=2)
    axs[1, 1].plot(x_tv, label=f"Total Variation Prior (Err: {err_tv:.2f})", color="green", linewidth=2)
    axs[1, 1].set_title("Physics-Informed Edge-Preserving Reconstruction (TV)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("metrology_inversion_benchmark.png", dpi=300)
    print("[*] Benchmark plot saved to: metrology_inversion_benchmark.png")

if __name__ == "__main__":
    main()
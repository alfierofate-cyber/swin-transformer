#!/usr/bin/env python3
"""
Generate iteration count comparison bar chart (Figure 4 style from paper)
Runs actual experiments on RHD3D-1T and RHD3D-3T problems
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
n: int = 128  # Grid size
tol_res: float = 1e-10  # Convergence threshold
max_iter: int = 20000

# Auto-detect available devices
# Note: MPS doesn't support float64, so we use CPU on Mac systems
if torch.cuda.is_available():
    device_type = "cuda"
    num_devices = torch.cuda.device_count()
else:
    # Use CPU for Mac and other systems (required for float64 precision)
    device_type = "cpu"
    num_devices = os.cpu_count() or 1

# Limit parallel workers to avoid memory issues
num_workers = min(num_devices, 4)

# ============================================================================
# Matrix Generation Functions (RHD Equations)
# ============================================================================
def make_kappa(case_name: str, n: int, device: str, dtype: torch.dtype, seed: int = 42):
    """Generate coefficient field for RHD equations"""
    g = torch.Generator(device=device).manual_seed(seed)

    # RHD3D-1T: Single-temperature radiation hydrodynamics (weak multiscale)
    if case_name.startswith("RHD3D-1T"):
        # 99.85% of rows in [1,10) interval, 0.15% can reach ~20
        base_coeff = 1.0
        perturbation = 1.0 + 8.0 * torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        mask = torch.rand((n, n, n), device=device, dtype=dtype, generator=g) > 0.9985
        perturbation[mask] = perturbation[mask] * 2.0
        k = base_coeff * perturbation
        return k, k, k

    # RHD3D-3T: Three-temperature model (strong multiscale)
    if case_name.startswith("RHD3D-3T"):
        delta = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        base = torch.rand((n, n, n), device=device, dtype=dtype, generator=g)
        base = base * 9.0 + 1.0  # [1, 10)

        k = torch.ones_like(delta)
        k[delta < 0.0824] = base[delta < 0.0824] * 1.5  # [1, 10)

        mask = (delta >= 0.0824) & (delta < 0.0824 + 0.0206)
        k[mask] = base[mask] * 15.0  # [1e1, 1e2)

        mask = (delta >= 0.0824 + 0.0206) & (delta < 0.0824 + 0.0206 + 0.0206)
        k[mask] = base[mask] * 150.0  # [1e2, 1e3)

        # Continue for higher scales up to 1e18
        mask = (delta >= 0.0824 + 3*0.0206) & (delta < 0.0824 + 4*0.0206)
        k[mask] = base[mask] * 1.5e4  # [1e4, 1e5)

        mask = (delta >= 0.0824 + 4*0.0206) & (delta < 0.0824 + 5*0.0206)
        k[mask] = base[mask] * 1.5e5  # [1e5, 1e6)

        mask = (delta >= 0.0824 + 5*0.0206) & (delta < 0.0824 + 6*0.0206)
        k[mask] = base[mask] * 1.5e6  # [1e6, 1e7)

        mask = delta >= 0.0824 + 6*0.0206
        k[mask] = base[mask] * 1.5e17  # [1e17, 1e18)

        return k, k, k

    raise ValueError(f"Unknown case: {case_name}")

def build_A_mv(kx, ky, kz, n: int, h: float, device: str):
    """Build matrix-vector product function for 7-point stencil"""
    def matvec(x):
        x3d = x.view(n, n, n)
        out = torch.zeros_like(x3d)
        h2 = h * h

        # Interior points
        out[1:-1, 1:-1, 1:-1] = (
            (kx[1:-1, 1:-1, 1:-1] + kx[2:, 1:-1, 1:-1]) * (x3d[2:, 1:-1, 1:-1] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (kx[1:-1, 1:-1, 1:-1] + kx[:-2, 1:-1, 1:-1]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[:-2, 1:-1, 1:-1]) / h2 +
            (ky[1:-1, 1:-1, 1:-1] + ky[1:-1, 2:, 1:-1]) * (x3d[1:-1, 2:, 1:-1] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (ky[1:-1, 1:-1, 1:-1] + ky[1:-1, :-2, 1:-1]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[1:-1, :-2, 1:-1]) / h2 +
            (kz[1:-1, 1:-1, 1:-1] + kz[1:-1, 1:-1, 2:]) * (x3d[1:-1, 1:-1, 2:] - x3d[1:-1, 1:-1, 1:-1]) / h2 -
            (kz[1:-1, 1:-1, 1:-1] + kz[1:-1, 1:-1, :-2]) * (x3d[1:-1, 1:-1, 1:-1] - x3d[1:-1, 1:-1, :-2]) / h2
        )
        return -out.view(-1)

    return matvec

def block_jacobi_preconditioner(kx, ky, kz, n: int, h: float, nb: int, device: str, dtype_prec: torch.dtype):
    """Build block-Jacobi preconditioner"""
    num_blocks_per_dim = n // nb
    h2 = h * h

    # Pad coefficients for boundary access
    kx_pad = torch.zeros((n+1, n, n), device=device, dtype=kx.dtype)
    ky_pad = torch.zeros((n, n+1, n), device=device, dtype=ky.dtype)
    kz_pad = torch.zeros((n, n, n+1), device=device, dtype=kz.dtype)

    kx_pad[:-1, :, :] = kx
    kx_pad[-1, :, :] = kx[-1, :, :]
    ky_pad[:, :-1, :] = ky
    ky_pad[:, -1, :] = ky[:, -1, :]
    kz_pad[:, :, :-1] = kz
    kz_pad[:, :, -1] = kz[:, :, -1]

    blocks_inv = []
    for i in range(num_blocks_per_dim):
        for j in range(num_blocks_per_dim):
            for k in range(num_blocks_per_dim):
                i_start, i_end = i * nb, (i + 1) * nb
                j_start, j_end = j * nb, (j + 1) * nb
                k_start, k_end = k * nb, (k + 1) * nb

                local_kx = kx_pad[i_start:i_end+1, j_start:j_end, k_start:k_end].to(dtype_prec)
                local_ky = ky_pad[i_start:i_end, j_start:j_end+1, k_start:k_end].to(dtype_prec)
                local_kz = kz_pad[i_start:i_end, j_start:j_end, k_start:k_end+1].to(dtype_prec)

                D_block = torch.zeros((nb, nb, nb), device=device, dtype=dtype_prec)
                for ii in range(nb):
                    for jj in range(nb):
                        for kk in range(nb):
                            D_block[ii, jj, kk] = (
                                (local_kx[ii, jj, kk] + local_kx[ii+1, jj, kk]) / h2 +
                                (local_ky[ii, jj, kk] + local_ky[ii, jj+1, kk]) / h2 +
                                (local_kz[ii, jj, kk] + local_kz[ii, jj, kk+1]) / h2
                            )

                D_block_inv = 1.0 / D_block
                blocks_inv.append(D_block_inv.to(torch.float32))

    def apply_preconditioner(r):
        r3d = r.view(n, n, n)
        z3d = torch.zeros_like(r3d)
        block_idx = 0
        for i in range(num_blocks_per_dim):
            for j in range(num_blocks_per_dim):
                for k in range(num_blocks_per_dim):
                    i_start, i_end = i * nb, (i + 1) * nb
                    j_start, j_end = j * nb, (j + 1) * nb
                    k_start, k_end = k * nb, (k + 1) * nb
                    z3d[i_start:i_end, j_start:j_end, k_start:k_end] = (
                        r3d[i_start:i_end, j_start:j_end, k_start:k_end] * blocks_inv[block_idx]
                    )
                    block_idx += 1
        return z3d.view(-1)

    return apply_preconditioner

# ============================================================================
# PCG Solver
# ============================================================================
def pcg_solve(A_mv, M_inv, b, x0, tol: float, max_iter: int, adp_tol: float = None, mode: str = "hl"):
    """Preconditioned Conjugate Gradient solver"""
    x = x0.clone()
    r = b - A_mv(x)
    z = M_inv(r)
    p = z.clone()

    rz_old = torch.dot(r, z)

    for it in range(max_iter):
        Ap = A_mv(p)
        alpha = rz_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        # Check convergence
        res_norm = torch.norm(r).item()
        if res_norm < tol:
            return x, it + 1

        # Adaptive precision switching
        if adp_tol is not None:
            if mode == "hl" and res_norm < adp_tol:
                z = M_inv(r)
            elif mode == "lh" and res_norm >= adp_tol:
                z = M_inv(r)
            else:
                z = M_inv(r)
        else:
            z = M_inv(r)

        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x, max_iter

# ============================================================================
# Single Configuration Solver
# ============================================================================
def solve_single_config(args):
    """Solve a single configuration and return iteration count"""
    device_id, problem, method, adp_tol_val, mode = args

    # Set device based on available hardware
    if device_type == "cuda":
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"

    try:
        # Generate problem
        kx, ky, kz = make_kappa(problem, n, device, torch.float64, seed=42)
        h = 1.0 / n

        # Build matrix-vector product
        A_mv = build_A_mv(kx, ky, kz, n, h, device)

        # Generate RHS
        g = torch.Generator(device=device).manual_seed(123)
        b = torch.randn(n*n*n, device=device, dtype=torch.float64, generator=g)
        x0 = torch.zeros_like(b)

        # Determine preconditioner precision
        if "fp64" in method or "fp80" in method:
            dtype_prec = torch.float64
        elif "fp32" in method:
            dtype_prec = torch.float32
        else:
            dtype_prec = torch.float64

        # Build preconditioner
        nb = 32
        M_inv = block_jacobi_preconditioner(kx, ky, kz, n, h, nb, device, dtype_prec)

        # Solve
        if "aMP" in method:
            _, iters = pcg_solve(A_mv, M_inv, b, x0, tol_res, max_iter, adp_tol_val, mode)
        else:
            _, iters = pcg_solve(A_mv, M_inv, b, x0, tol_res, max_iter)

        result = (problem, method, iters)

    finally:
        # Clean up memory
        if 'kx' in locals():
            del kx, ky, kz
        if 'b' in locals():
            del b, x0
        if device_type == "cuda":
            torch.cuda.empty_cache()

    return result

# ============================================================================
# Main Execution
# ============================================================================
def collect_iteration_data():
    """Collect iteration counts for all method-problem combinations"""

    problems = ["RHD3D-1T", "RHD3D-3T"]

    # Define test configurations
    tasks = []
    gpu_id = 0

    # For each problem
    for problem in problems:
        # fp64-uniform (baseline)
        tasks.append((gpu_id % num_workers, problem, "fp64-uniform", None, None))
        gpu_id += 1

        # fp32-fMP (fixed mixed precision)
        tasks.append((gpu_id % num_workers, problem, "fp32-fMP", None, None))
        gpu_id += 1

        # fp32-aMP-BJAC(hl) with different tolerances
        for adp_tol in [10.0, 1.0, 0.1]:
            tasks.append((gpu_id % num_workers, problem, f"fp32-aMP(hl)-{adp_tol}", adp_tol, "hl"))
            gpu_id += 1

        # fp32-aMP-BJAC(lh) with different tolerances
        for adp_tol in [10.0, 1.0, 0.1]:
            tasks.append((gpu_id % num_workers, problem, f"fp32-aMP(lh)-{adp_tol}", adp_tol, "lh"))
            gpu_id += 1

    print(f"Running {len(tasks)} experiments on {num_workers} workers ({device_type})...")

    # Run in parallel - limit to num_workers processes to prevent memory overflow
    # Use chunksize=1 to ensure tasks are distributed one at a time as workers become available
    with Pool(processes=num_workers) as pool:
        results = pool.map(solve_single_config, tasks, chunksize=1)

    return results

def plot_iteration_comparison(results: List[Tuple[str, str, int]], output_path: str = "out/iteration_comparison.png"):
    """Generate bar chart comparing iteration counts"""

    # Organize data
    data = {}
    for problem, method, iters in results:
        if problem not in data:
            data[problem] = {}
        data[problem][method] = iters

    # Define method display order and labels
    method_groups = {
        "fp64-uniform": ("fp64-uniform", "#1f77b4", ""),
        "fp32-fMP": ("fp32-fMP", "#ff7f0e", ""),
        "fp32-aMP(hl)-10.0": ("aMP(hl)-10", "#2ca02c", ""),
        "fp32-aMP(hl)-1.0": ("aMP(hl)-1", "#d62728", ""),
        "fp32-aMP(hl)-0.1": ("aMP(hl)-0.1", "#9467bd", ""),
        "fp32-aMP(lh)-10.0": ("aMP(lh)-10", "#8c564b", "//"),
        "fp32-aMP(lh)-1.0": ("aMP(lh)-1", "#e377c2", "//"),
        "fp32-aMP(lh)-0.1": ("aMP(lh)-0.1", "#7f7f7f", "//"),
    }

    problems = sorted(data.keys())
    methods = list(method_groups.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(problems))
    width = 0.1

    # Plot bars
    for i, method in enumerate(methods):
        label, color, hatch = method_groups[method]
        iters = [data[prob].get(method, 0) for prob in problems]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, iters, width, label=label, color=color,
                     edgecolor='black', linewidth=0.8, hatch=hatch)

        # Add value labels on top of bars
        for bar, iter_val in zip(bars, iters):
            if iter_val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(iter_val)}',
                       ha='center', va='bottom', fontsize=8, rotation=0)

    # Formatting
    ax.set_xlabel('Problem', fontsize=12, fontweight='bold')
    ax.set_ylabel('Iteration Count', fontsize=12, fontweight='bold')
    ax.set_title('Iteration Comparison: RHD3D Problems (tol=1e-10)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problems, fontsize=11)
    ax.legend(ncol=4, loc='upper left', fontsize=9, frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure to: {output_path}")
    plt.show()

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generating Iteration Comparison (Figure 4 Style)")
    print("=" * 70)
    print(f"Grid size: {n}x{n}x{n}")
    print(f"Convergence threshold: {tol_res}")
    print(f"Max iterations: {max_iter}")
    print(f"Device type: {device_type}")
    print(f"Number of workers: {num_workers}")
    print("=" * 70)

    # Collect data
    results = collect_iteration_data()

    # Print results
    print("\n" + "=" * 70)
    print("Iteration Count Results:")
    print("=" * 70)
    for problem, method, iters in sorted(results):
        print(f"{problem:15s} | {method:25s} | {iters:5d} iterations")

    # Generate plot
    plot_iteration_comparison(results)

    print("\n" + "=" * 70)
    print("✓ Complete!")
    print("=" * 70)

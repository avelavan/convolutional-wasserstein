import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOL = 1e-5
N_STEPS = 20
N = 200
SIGMA = 0.05
SIGMA2 = SIGMA ** 2  # 0.0025

means = [[0.25, 0.75], [0.75, 0.25]]
alphas = [0.5, 0.5]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(OUT_DIR, "debiased_barycenter_error.npz")
PNG_PATH = os.path.join(OUT_DIR, "debiased_barycenter_error.png")


def make_mus(V):
    from firedrake import Function, SpatialCoordinate, assemble, dx, exp, pi
    x, y = SpatialCoordinate(V.mesh())
    mus = []
    for mean in means:
        f = Function(V)
        f.interpolate(
            (1 / (2 * pi * SIGMA ** 2))
            * exp(-((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / (2 * SIGMA ** 2))
        )
        f.assign(f / assemble(f * dx))
        mus.append(f)
    return mus


def run_sweep():
    from firedrake import FunctionSpace, UnitSquareMesh
    from barycenter import debiased_wasserstein_barycenter, gaussian_stats

    V = FunctionSpace(UnitSquareMesh(N, N), "CG", 1)
    mus = make_mus(V)

    epsilons = np.geomspace(0.1, 0.002, 12)

    ratios = []
    for eps in epsilons:
        print("=" * 60)
        print(f"epsilon = {eps:.5f}")
        print("=" * 60)
        t0 = time.perf_counter()
        bary, _, _ = debiased_wasserstein_barycenter(
            mus, alphas, V, epsilon=float(eps), tol=TOL, n_steps=N_STEPS
        )
        print(f"Wall time: {time.perf_counter() - t0:.2f}s")
        _, cov = gaussian_stats(bary, label=f"eps={eps:.4f}")
        ratios.append(cov[0, 0] / SIGMA2)
        print(f"  cov[0,0]/sigma^2 = {ratios[-1]:.6f}")

    np.savez(NPZ_PATH, epsilons=epsilons, ratios=np.array(ratios))
    print(f"\nSaved data to {NPZ_PATH}")


def plot_from_npz():
    data = np.load(NPZ_PATH)
    epsilons = data["epsilons"]
    ratios = data["ratios"]

    # Reconstruct Σ̂ from the saved scalar by copying cov[0,0] into the
    # bottom-right and zeroing off-diagonals: Σ̂ = a·I, a = ratios·σ².
    # True barycenter: N((0.5,0.5), σ²I). Means equal, so ‖·‖_{L²(ℝ²)}
    # depends only on covariances. For isotropic 2D Gaussians g_a, g_b:
    #   ‖g_a − g_b‖² = 1/(4π a) + 1/(4π b) − 1/(π (a+b))
    a = ratios * SIGMA2
    b = SIGMA2
    norm_sq = 1.0 / (4 * np.pi * a) + 1.0 / (4 * np.pi * b) - 1.0 / (np.pi * (a + b))
    norm = np.sqrt(np.maximum(norm_sq, 0.0))

    num_points = -6
    log_eps = np.log(epsilons)
    log_norm = np.log(norm)
    eps_tail = epsilons[num_points:]
    log_eps_tail = log_eps[num_points:]
    log_norm_tail = log_norm[num_points:]
    slope, intercept = np.polyfit(log_eps_tail, log_norm_tail, 1)
    fit_line = np.exp(intercept) * eps_tail ** slope
    print(f"Fit (last {-num_points}): error ≈ {np.exp(intercept):.4g} * eps^{slope:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, norm, marker="o",
            label=r"$\|\hat g - g_{\mathrm{true}}\|_{L^2(\mathbb{R}^2)}$")
    ax.plot(eps_tail, fit_line, "--",
            label=fr"fit (last {-num_points}): slope={slope:.2f}, $R^2$={r2:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\|\hat g - g_{\mathrm{true}}\|_{L^2(\mathbb{R}^2)}$")
    ax.set_title("Reconstructed Gaussian barycenter error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"Saved plot to {PNG_PATH}")


if __name__ == "__main__":
    plot_from_npz()

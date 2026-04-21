import time
import numpy as np
import matplotlib.pyplot as plt
from firedrake import (
    assemble,
    conditional,
    ln,
    dx,
    Function,
    interpolate,
    norm,
    FunctionSpace,
    UnitSquareMesh,
    SpatialCoordinate,
    pi,
    exp,
)
from firedrake.pyplot import tripcolor
from solvers import BackwardEuler
from scipy.optimize import root_scalar


def gaussian_stats(mu, label=""):
    """
    Compute and print the mean and covariance of a distribution,
    assuming it is Gaussian, using numerical integration.
    """
    mesh = mu.function_space().mesh()
    x, y = SpatialCoordinate(mesh)
    mean_x = assemble(x * mu * dx)
    mean_y = assemble(y * mu * dx)
    cov_xx = assemble((x - mean_x) ** 2 * mu * dx)
    cov_xy = assemble((x - mean_x) * (y - mean_y) * mu * dx)
    cov_yy = assemble((y - mean_y) ** 2 * mu * dx)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    header = f"Gaussian stats [{label}]" if label else "Gaussian stats"
    print(f"\n{header}")
    print(f"  Mean:       ({mean_x:.6f}, {mean_y:.6f})")
    print(f"  Covariance: [[{cov_xx:.6f}, {cov_xy:.6f}],")
    print(f"               [{cov_xy:.6f}, {cov_yy:.6f}]]")
    return np.array([mean_x, mean_y]), cov


def _entropy(mu):
    """
    Computes entropy of a probability distributions.

    Args:
        mu: probability distribution
    Returns:
        entropy: scalar entropy value
    """
    entropy = -1 * assemble(mu * ln(mu + 1e-12) * dx)
    return entropy


def _find_beta(mu, h0, tol=1e-5, maxiter=200):
    """
    Performs root-finding to find beta value

    Args:
        mu: unsharpened barycenter
        h0: user defined parameter, from _entropic_sharpening
        tol: tolerance parameter for scipy solver
        maxiter: maximum iterations before defaulting to beta=1

    Returns:
        beta: scalar solution to equation
    """
    V = mu.function_space()
    tmp = Function(V)

    def objective(beta):
        tmp.interpolate(mu ** beta)
        tmp.interpolate(tmp / assemble(tmp * dx))
        return _entropy(tmp) - h0

    a, b = 1.0, 2.0
    try:
        if objective(b) > 0:
            b = 5.0
        result = root_scalar(objective, bracket=[a, b], method='brentq',
                             xtol=tol, maxiter=maxiter)
        return result.root if result.converged else 1.0
    except ValueError:
        return 1.0


def _entropic_sharpening(mu, h0):
    """
    Adds entropic sharpening for computation of Wasserstein Barycenter.

    Args:
        mu: unsharpened Wasserstein barycenter
        a: list of weights associated to corresponding mus, sum(a) = 1
        h0: user defined parameter, usually max H(mu_i)
    Returns:
        sharp_mu: sharpened barycenter
    """

    h_mu = _entropy(mu)
    if h_mu <= h0:
        return mu

    print("sharpening!")
    beta = _find_beta(mu, h0)
    if beta == 1.0:
        return mu
    print(f"beta = {beta}, mass = {assemble(mu*dx)}")
    mu.interpolate(mu ** beta)
    mu.interpolate(mu / assemble(mu * dx))
    return mu


def wasserstein_barycenter(
    mus, alphas, V, epsilon=0.05, tol=1e-7, maxiter=100, v=None, w=None, sharpen=True,
    n_steps=1,
):
    """
    Compute the Wasserstein barycenter of given distributions at a single mesh level.

    Supports epsilon warm-starting: if v/w are provided from a previous
    run at a different epsilon, the scaling vectors are rescaled accordingly.
    """
    num_dists = len(mus)
    if abs(sum(alphas) - 1) > 1e-10:
        raise ValueError(f"Weights must sum to 1, got {sum(alphas)}")

    mu = Function(V, name="mu").assign(1.0)
    mu.interpolate(mu / assemble(mu * dx))
    d = []

    if v is None and w is None:
        if n_steps == 1:
            v = [BackwardEuler(V, dt=epsilon / 2) for _ in range(num_dists)]
            w = [BackwardEuler(V, dt=epsilon / 2) for _ in range(num_dists)]
        else:
            v = [BackwardEuler(V, dt=epsilon / (2 * n_steps), n_steps=n_steps) for _ in range(num_dists)]
            w = [BackwardEuler(V, dt=epsilon / (2 * n_steps), n_steps=n_steps) for _ in range(num_dists)]
        for i in range(num_dists):
            v[i].initialise()
            w[i].initialise()
    else:
        for i in range(num_dists):
            old_epsilon = 2 * n_steps * float(getattr(v[i], "_total_dt", v[i].dt_const))
            ratio = old_epsilon / epsilon
            v[i].rhs.interpolate(v[i].rhs ** ratio)
            v[i].update_dt(epsilon / (2 * n_steps))
            w[i].update_dt(epsilon / (2 * n_steps))

    for _ in range(num_dists):
        d.append(Function(V).assign(1.0))

    curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

    j = 0
    res = float("inf")
    res_best = float("inf")
    stall_count = 0
    stall_patience = 5
    stall_min_improvement = 1e-2
    mu_prev = Function(V)

    while (res > tol) and (j < maxiter):
        mu_prev.assign(mu)
        mu.assign(1.0)
        # NOTE: this loop has sequential dependencies on mu — parallelising requires
        # a Jacobi-style update (compute all d[i] from previous mu, then update mu)
        for i in range(num_dists):
            v[i].solve()
            w[i].update(curr[i] / v[i].output_function)
            w[i].solve()
            d[i].interpolate(v[i].rhs * w[i].output_function)
            mu.interpolate(mu * (d[i] ** alphas[i]))

        # mu.interpolate(conditional(mu < 1e-12, 1e-12, mu))
        mu.interpolate(mu / assemble(mu * dx))  # normalise before sharpening so entropy is comparable
        h0 = max(map(_entropy, mus))
        if sharpen:
            mu = _entropic_sharpening(mu, h0)

        for i in range(num_dists):
            v[i].update(v[i].rhs * (mu / d[i]))

        res = norm(mu - mu_prev)
        print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
        j += 1

        if res < res_best * (1.0 - stall_min_improvement):
            res_best = res
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= stall_patience:
                print(f"  early stop: residual stalled for {stall_patience} iters (best={res_best:.6e})")
                break

    return mu, v, w


# ── Setup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EPSILON_TARGET = 0.001
    TOL = 1e-5
    N = 200  # single fine mesh

    V = FunctionSpace(UnitSquareMesh(N, N), "CG", 1)

    # Define input Gaussians
    means = [[0.25, 0.25], [0.75, 0.75]]
    sigma = 0.05
    x, y = SpatialCoordinate(V.mesh())

    mus = []
    for mean in means:
        f = Function(V)
        f.interpolate(
            (1 / (2 * pi * sigma**2))
            * exp(-((x - mean[0]) ** 2 + (y - mean[1]) ** 2) / (2 * sigma**2))
        )
        f.assign(f / assemble(f * dx))
        mus.append(f)

    alphas = [0.5, 0.5]

    # ── Experiment ─────────────────────────────────────────────────────────────

    print("=" * 60)
    print(f"Mesh: {N}x{N},  target eps={EPSILON_TARGET},  tol={TOL}")
    print("=" * 60)

    N_STEPS = 5

    print(f"\n[A] Single-step Euler — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_single, _, _ = wasserstein_barycenter(
        mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL, sharpen=True, n_steps=1
    )
    t_single = time.perf_counter() - t0
    print(f"Wall time: {t_single:.2f}s")
    gaussian_stats(bary_single, label="A: single-step Euler")

    print(f"\n[B] Multi-step Euler ({N_STEPS} sub-steps) — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_multi, _, _ = wasserstein_barycenter(
        mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL, sharpen=True, n_steps=N_STEPS
    )
    t_multi = time.perf_counter() - t0
    print(f"Wall time: {t_multi:.2f}s")
    gaussian_stats(bary_multi, label="B: multi-step Euler")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mu_init = Function(V).assign(0.0)
    for f in mus:
        mu_init.interpolate(mu_init + f)
    c_init = tripcolor(mu_init, axes=axes[0])
    fig.colorbar(c_init, ax=axes[0])
    axes[0].set_title("Initial distributions")
    axes[0].set_aspect("equal")

    c0 = tripcolor(bary_single, axes=axes[1])
    fig.colorbar(c0, ax=axes[1])
    axes[1].set_title(f"[A] Single-step Euler (eps={EPSILON_TARGET})")
    axes[1].set_aspect("equal")

    c1 = tripcolor(bary_multi, axes=axes[2])
    fig.colorbar(c1, ax=axes[2])
    axes[2].set_title(f"[B] {N_STEPS}-step Euler (eps={EPSILON_TARGET})")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.show()

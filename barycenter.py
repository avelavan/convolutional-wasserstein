import time
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor
from solvers import HeatEquationSolver
from scipy.optimize import root_scalar


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


def _find_beta(mu, h0, tol=1e-5, maxiter=50):
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
    pass


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
    mu_int = assemble(mu * dx)

    if h_mu + mu_int > h0 + 1:
        # root-finding here
        print("sharpening!")
        beta = 1.0
        mu.interpolate(mu**beta)
        sharp_mu = mu
    else:
        sharp_mu = mu  # beta = 1 as per paper

    return sharp_mu


def wasserstein_barycenter(
    mus, alphas, V, epsilon=0.05, tol=1e-7, maxiter=20, v=None, w=None
):
    """
    Compute the Wasserstein barycenter of given distributions at a single mesh level.

    Supports epsilon warm-starting: if v/w are provided from a previous
    run at a different epsilon, the scaling vectors are rescaled accordingly.
    """
    num_dists = len(mus)
    try:
        assert abs(sum(alphas)) == 1, "Weights must sum to 1."
    except AssertionError as e:
        print("Error in weights: ", e)
        raise e

    mu = Function(V, name="mu").assign(1.0)
    mu.interpolate(mu / assemble(mu * dx))
    w_prev = Function(V).assign(1.0)
    d = []

    if v is None and w is None:
        v = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
        w = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
        for i in range(num_dists):
            v[i].initialise()
            w[i].initialise()
    else:
        for i in range(num_dists):
            old_epsilon = 2 * float(v[i].dt_const)
            ratio = old_epsilon / epsilon
            v[i].rhs.interpolate(v[i].rhs ** ratio)
            v[i].update_dt(epsilon / 2)
            w[i].update_dt(epsilon / 2)

    for _ in range(num_dists):
        d.append(Function(V).assign(1.0))

    curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

    j = 0
    res = 1
    while (res > tol) and (j < maxiter):
        mu.assign(1.0)
        # NOTE: this loop has sequential dependencies on mu — parallelising requires
        # a Jacobi-style update (compute all d[i] from previous mu, then update mu)
        res = 0
        for i in range(num_dists):
            w_prev.assign(w[i].rhs)
            v[i].solve()  # application of the heat kernel?
            w[i].update(curr[i] / v[i].output_function)
            w[i].solve()
            d[i].interpolate(v[i].rhs * w[i].output_function)
            mu.interpolate(mu * (d[i] ** alphas[i]))
            res = max(norm(w_prev - w[i].rhs), res)

        h0 = max(map(_entropy, mus))  # as defined in paper
        mu = _entropic_sharpening(mu, h0)

        for i in range(num_dists):
            v[i].update(v[i].rhs * (mu / d[i]))
        print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
        j += 1

    return mu, v, w


# ── Setup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EPSILON_TARGET = 0.01
    TOL = 1e-5
    N = 200  # single fine mesh

    V = FunctionSpace(UnitSquareMesh(N, N), "CG", 1)

    # Define input Gaussians
    means = [[0.4, 0.4], [0.6, 0.6]]
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

    print(f"\nCold start — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_cold, _, _ = wasserstein_barycenter(
        mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL
    )
    t_cold = time.perf_counter() - t0
    print(f"Wall time: {t_cold:.2f}s")
    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))

    colors1 = tripcolor(bary_cold, axes=axes)
    fig.colorbar(colors1, ax=axes)
    axes.set_title(f"[A] Cold start (eps={EPSILON_TARGET})")

    plt.show()
    """
    colors2 = tripcolor(bary_sched, axes=axes[1])
    fig.colorbar(colors2, ax=axes[1])
    axes[1].set_title(f"[B] Eps schedule (final eps={EPSILON_TARGET})")

    plt.tight_layout()
    plt.show()
    """

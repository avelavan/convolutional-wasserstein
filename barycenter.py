import time
import matplotlib.pyplot as plt
from firedrake import (
    assemble,
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

def _entropic_limit_update(mu, h0):
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
    if h_mu <= h0:
        return mu

    print("sharpening!")
    beta = _find_beta(mu, h0)
    mu.interpolate(mu ** beta)
    return mu


def wasserstein_barycenter(
    mus, alphas, V, epsilon=0.05, tol=1e-7, maxiter=100, v=None, w=None, sharpen=True
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
        if sharpen:
            mu = _entropic_sharpening(mu, h0)
        mu.interpolate(mu / assemble(mu * dx)) # normalisation step

        for i in range(num_dists):
            v[i].update(v[i].rhs * (mu / d[i]))
        print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
        j += 1

    return mu, v, w


# ── Setup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EPSILON_TARGET = 0.001
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

    print(f"\n[A] Without sharpening — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_no_sharp, _, _ = wasserstein_barycenter(
        mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL, sharpen=False
    )
    t_no_sharp = time.perf_counter() - t0
    print(f"Wall time: {t_no_sharp:.2f}s")

    print(f"\n[B] With sharpening — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_sharp, _, _ = wasserstein_barycenter(
        mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL, sharpen=True
    )
    t_sharp = time.perf_counter() - t0
    print(f"Wall time: {t_sharp:.2f}s")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    c0 = tripcolor(bary_no_sharp, axes=axes[0])
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_title(f"[A] No sharpening (eps={EPSILON_TARGET})")

    c1 = tripcolor(bary_sharp, axes=axes[1])
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_title(f"[B] With sharpening (eps={EPSILON_TARGET})")

    plt.tight_layout()
    plt.show()

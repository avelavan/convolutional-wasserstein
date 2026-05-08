import time
import numpy as np
import matplotlib.pyplot as plt
import math
from firedrake import *
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
    Computes entropy safely, surviving CG2 negative undershoots.
    """

    # Clamp both to ensure the negative ripples contribute 0 to the entropy
    # rather than crashing the logarithm
    # safe_mu = max_value(mu, 1e-12)
    integrand = conditional(mu > 0, mu * ln(mu), 0)
    entropy = -1 * assemble(integrand * dx)

    return entropy

def _find_beta(mu, h0, tol=1e-4, maxiter=50):

    V = mu.function_space()
    tmp = Function(V)
    beta_c = Constant(1.0)

    # Safe base for exponentiation
    # safe_mu = max_value(mu, 1e-10)
    safe_mu = mu
    expr = safe_mu ** beta_c

    def objective(b_val):
        beta_c.assign(b_val)
        tmp.interpolate(expr)

        Z = assemble(tmp * dx)
        # Trap mass destruction
        # if Z < 1e-14 or math.isnan(Z):
            # return 999.0

        tmp.interpolate(tmp / Z)

        safe_tmp = max_value(tmp, 1e-12)
        safe_tmp = tmp
        ent = -1 * assemble(tmp * ln(safe_tmp) * dx)

        # Trap the NaN so bisection never collapses again
        '''
        if math.isnan(ent):
            return 999.0
        '''

        return ent - h0

    # Find a valid upper bound
    a = 1.0
    b = 2.0
    f_b = objective(b)

    iters = 0
    while f_b > 0 and iters < 3:
        b *= 2.0
        f_b = objective(b)
        iters += 1

    if f_b > 0:
        print(f"  [Warning] Could not perfectly bound beta. Max tried: {b:.2f}")
        return b

    # Bisection TODO: replace w Newton Method
    for _ in range(maxiter):
        mid = (a + b) / 2.0
        f_mid = objective(mid)

        if abs(f_mid) < tol:
            return mid

        # Normal Python logic that is now safe from NaNs
        if f_mid > 0:
            a = mid
        else:
            b = mid

    return (a + b) / 2.0


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
    # print(f"sharpening! {h_mu} > {h0}")
    beta = _find_beta(mu, h0)
    if beta == 1.0:
        return mu
    print(f"beta = {beta}, mass = {assemble(mu*dx)}")
    mu.interpolate(mu ** beta)
    mass = assemble(mu * dx)

    # normalise after each sharpening
    mu.interpolate(mu / mass)
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
    h0 = max(map(_entropy, mus))  # mus are fixed; hoist out of the loop

    while (res > tol) and (j < maxiter):
        mu_prev.assign(mu)
        mu.assign(1.0)
        # NOTE: this loop has sequential dependencies on mu — parallelising requires
        # a Jacobi-style update (compute all d[i] from previous mu, then update mu)
        for i in range(num_dists):
            v[i].solve()
            w[i].update(curr[i] / (v[i].output_function))
            w[i].solve()
            d[i].interpolate(v[i].rhs * w[i].output_function)
            mu.interpolate(mu * (d[i] ** alphas[i]))

        mu.interpolate(mu / assemble(mu * dx)) # normalise before sharpening so entropy is comparable
        if sharpen:
            mu = _entropic_sharpening(mu, h0)

        for i in range(num_dists):
            v[i].update(v[i].rhs * (mu / (d[i])))

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

def debiased_wasserstein_barycenter(
    mus, alphas, V, epsilon=0.05, tol=1e-7, maxiter=100, v=None, w=None,
    n_steps=5,
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
            p = BackwardEuler(V, dt=epsilon / 2)
        else:
            v = [BackwardEuler(V, dt=epsilon / (2 * n_steps), n_steps=n_steps) for _ in range(num_dists)]
            w = [BackwardEuler(V, dt=epsilon / (2 * n_steps), n_steps=n_steps) for _ in range(num_dists)]
            p = BackwardEuler(V, dt=epsilon / (2 * n_steps), n_steps=n_steps)
        for i in range(num_dists):
            v[i].initialise()
            w[i].initialise()
        p.initialise() # debiasing vector
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
    # h0 = max(map(_entropy, mus))  # mus are fixed; hoist out of the loop

    while (res > tol) and (j < maxiter):
        mu_prev.assign(mu)
        mu.assign(1.0)
        # NOTE: this loop has sequential dependencies on mu — parallelising requires
        # a Jacobi-style update (compute all d[i] from previous mu, then update mu)
        for i in range(num_dists):
            v[i].solve()
            w[i].update(curr[i] / (v[i].output_function))
            w[i].solve()
            d[i].interpolate(w[i].output_function)
            mu.interpolate(mu * (d[i] ** alphas[i]))

        mu.interpolate(mu * p.rhs)


        mu.interpolate(mu / assemble(mu * dx))

        p.solve()
        p.update(sqrt(p.rhs * mu / p.output_function))

        for i in range(num_dists):
            v[i].update((mu / (d[i])))

        # res = assemble(abs(mu - mu_prev)*dx) / assemble(abs(mu_prev)*dx)
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
    EPSILON_TARGET = 0.002
    TOL = 1e-5
    N_STEPS = 20

    means = [[0.25, 0.75], [0.75, 0.25]]
    sigma = 0.05
    alphas = [0.5, 0.5]

    def make_mus(V):
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
        return mus

    # CG(1) — low-order mesh
    N1 = 200
    V1 = FunctionSpace(UnitSquareMesh(N1, N1), "CG", 1)
    mus1 = make_mus(V1)

    x, y = SpatialCoordinate(V1.mesh())
    for i, m in enumerate(mus1):
        mx = assemble(x * m * dx)
        my = assemble(y * m * dx)
        vxx = assemble((x - mx) ** 2 * m * dx)
        vyy = assemble((y - my) ** 2 * m * dx)
        cxy = assemble((x - mx) * (y - my) * m * dx)
        total = assemble(m * dx)
        print(f"input {i}: mass={total:.6f} mean=({mx:.4f},{my:.4f}) "
              f"var=({vxx:.6f},{vyy:.6f}) cov={cxy:.6f}")

    # CG(2) — higher-order mesh (similar DOF count to 200x200 CG1)
    N2 = 200
    V2 = FunctionSpace(UnitSquareMesh(N2, N2), "CG", 1)
    mus2 = make_mus(V2)

    # ── Experiments ────────────────────────────────────────────────────────────

    print("=" * 60)
    print(f"target eps={EPSILON_TARGET},  tol={TOL},  sub-steps={N_STEPS}")
    print("=" * 60)

    print(f"\n[A] CG(1) {N1}x{N1} — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_lo, _, _ = debiased_wasserstein_barycenter(
        mus1, alphas, V1, epsilon=EPSILON_TARGET, tol=TOL, n_steps=N_STEPS
    )
    t_lo = time.perf_counter() - t0
    print(f"Wall time: {t_lo:.2f}s")
    gaussian_stats(bary_lo, label="A: CG(1) - debiased sinkhorn")

    print(f"\n[B] CG(2) {N2}x{N2} — eps={EPSILON_TARGET}")
    t0 = time.perf_counter()
    bary_hi, _, _ = wasserstein_barycenter(
        mus2, alphas, V2, epsilon=EPSILON_TARGET, tol=TOL, sharpen=True, n_steps=N_STEPS
    )
    t_hi = time.perf_counter() - t0
    print(f"Wall time: {t_hi:.2f}s")
    gaussian_stats(bary_hi, label="B: CG(1) - entropic sharpening")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mu_init = Function(V1).assign(0.0)
    for f in mus1:
        mu_init.interpolate(mu_init + f)
    c_init = tripcolor(mu_init, axes=axes[0])
    fig.colorbar(c_init, ax=axes[0])
    axes[0].set_title("Initial distributions")
    axes[0].set_aspect("equal")

    c_lo = tripcolor(bary_lo, axes=axes[1])
    fig.colorbar(c_lo, ax=axes[1])
    axes[1].set_title(f"[A] CG(1) {N1}x{N1} (eps={EPSILON_TARGET}) debiased sinkhorn")
    axes[1].set_aspect("equal")

    c_hi = tripcolor(bary_hi, axes=axes[2])
    fig.colorbar(c_hi, ax=axes[2])
    axes[2].set_title(f"[B] CG(1) {N2}x{N2} (eps={EPSILON_TARGET}) entropic sharpening")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.show()

import time

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import tripcolor

from solvers import HeatEquationSolver


def wasserstein_barycenter(
    mus, alphas, V, epsilon=0.05, tol=1e-7, maxiter=100, v_list=None, w_list=None
):
    """
    Compute the Wasserstein barycenter of given distributions at a single mesh level.

    Supports epsilon warm-starting: if v_list/w_list are provided from a previous
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
    test_func = Function(V).assign(1.0)
    d_list = []

    if v_list is None and w_list is None:
        v_list = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
        w_list = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
        for i in range(num_dists):
            v_list[i].initialise()
            w_list[i].initialise()
    else:
        for i in range(num_dists):
            old_epsilon = 2 * float(v_list[i].dt_const)
            ratio = old_epsilon / epsilon
            v_list[i].function.interpolate(v_list[i].function ** ratio)
            v_list[i].update_dt(epsilon / 2)
            w_list[i].update_dt(epsilon / 2)

    for _ in range(num_dists):
        d_list.append(Function(V).assign(1.0))

    curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

    j = 0
    res = 1
    while (res > tol) and (j < maxiter):
        mu.assign(1.0)
        # NOTE: this loop has sequential dependencies on mu — parallelising requires
        # a Jacobi-style update (compute all d_list[i] from previous mu, then update mu)
        res = 0
        for i in range(num_dists):
            test_func.assign(w_list[i].function)
            v_list[i].solve() # application of the heat kernel?
            w_list[i].update(curr[i] / v_list[i].output_function)
            w_list[i].solve()
            d_list[i].interpolate(v_list[i].function * w_list[i].output_function)
            mu.interpolate(mu * (d_list[i] ** alphas[i]))
            res = max(norm(test_func - w_list[i].function), res)
        for i in range(num_dists):
            v_list[i].update(v_list[i].function * (mu / d_list[i]))
        print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
        j += 1

    return mu, v_list, w_list


def multiscale_wasserstein_barycenter(
    mus, alphas, Vs, epsilons, tol=1e-7, coarse_tol=1e-2, maxiter=100
):
    """
    Compute the Wasserstein barycenter using a mesh hierarchy + epsilon schedule.

    At each level, scaling vectors are rescaled for the new epsilon and then
    interpolated onto the finer mesh, giving an accurate warm start for the
    next level. This makes the final (expensive) fine-mesh solve cheap.

    Parameters
    ----------
    mus        : Input distributions defined on Vs[-1] (finest mesh)
    alphas     : Barycenter weights, summing to 1
    Vs         : List of function spaces, coarse to fine
    epsilons   : Regularisation schedule, one value per level (same length as Vs)

    coarse_tol : Convergence tolerance for all coarser levels — looser is fine
                 since these levels only need to provide a good warm start
    maxiter    : Max Sinkhorn iterations per level
    """
    assert len(Vs) == len(epsilons), "Vs and epsilons must have the same length"
    num_dists = len(mus)
    v_list = None
    w_list = None
    total_iters = []

    for level, (V, epsilon) in enumerate(zip(Vs, epsilons)):
        level_tol = tol if level == len(Vs) - 1 else coarse_tol
        print(
            f"\n--- Level {level + 1}/{len(Vs)}: {V.mesh().num_cells()} cells, eps={epsilon:.4f} ---"
        )

        if v_list is None:
            # Cold start at coarsest level
            v_list = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
            w_list = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
            for i in range(num_dists):
                v_list[i].initialise()
                w_list[i].initialise()
        else:
            # Rescale to new epsilon, then interpolate onto the finer mesh
            for i in range(num_dists):
                old_epsilon = 2 * float(v_list[i].dt_const)
                ratio = old_epsilon / epsilon
                v_list[i].function.interpolate(v_list[i].function ** ratio)
                v_list[i].refine(V, epsilon / 2)
                w_list[i].refine(V, epsilon / 2)

        # Project input distributions to current mesh level
        curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

        # Sinkhorn loop at this level
        mu = Function(V, name="mu").assign(1.0)
        mu.interpolate(mu / assemble(mu * dx))
        test_func = Function(V).assign(1.0)
        d_list = [Function(V).assign(1.0) for _ in range(num_dists)]

        j, res = 0, 1.0
        while res > level_tol and j < maxiter:
            mu.assign(1.0)
            res = 0.0
            for i in range(num_dists):
                test_func.assign(w_list[i].function)
                v_list[i].solve()
                w_list[i].update(curr[i] / v_list[i].output_function)
                w_list[i].solve()
                d_list[i].interpolate(v_list[i].function * w_list[i].output_function)
                mu.interpolate(mu * (d_list[i] ** alphas[i]))
                res = max(norm(test_func - w_list[i].function), res)
            for i in range(num_dists):
                v_list[i].update(v_list[i].function * (mu / d_list[i]))
            print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
            j += 1

        total_iters.append(j)

    print(f"\nTotal iterations per level: {total_iters}  (sum={sum(total_iters)})")
    return mu, v_list, w_list


# ── Setup ──────────────────────────────────────────────────────────────────

EPSILON_TARGET = 0.00005
TOL = 1e-2
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

# Epsilon schedule: ratio ~1.3 between levels so rescaling stays accurate.
# Starting ~8x above target means early levels converge in 1-2 iterations.
EPSILON_SCHEDULE = [0.8, 0.6, 0.45, 0.34, 0.26, 0.2, 0.15, EPSILON_TARGET]

# ── Experiment ─────────────────────────────────────────────────────────────

print("=" * 60)
print(f"Mesh: {N}x{N},  target eps={EPSILON_TARGET},  tol={TOL}")
print(f"Epsilon schedule: {EPSILON_SCHEDULE}")
print("=" * 60)

print(f"\n[A] Cold start — eps={EPSILON_TARGET} directly")
t0 = time.perf_counter()
bary_cold, _, _ = wasserstein_barycenter(
    mus, alphas, V, epsilon=EPSILON_TARGET, tol=TOL
)
t_cold = time.perf_counter() - t0
print(f"Wall time: {t_cold:.2f}s")
"""
COARSE_TOL = 1e-6

print("\n[B] Epsilon schedule with warm-starting")
t0 = time.perf_counter()
v_list, w_list = None, None
for eps in EPSILON_SCHEDULE:
    level_tol = TOL if eps == EPSILON_SCHEDULE[-1] else COARSE_TOL
    print(f"\n  --- eps={eps:.2f} (tol={level_tol:.0e}) ---")
    _, v_list, w_list = wasserstein_barycenter(
        mus, alphas, V, epsilon=eps, tol=level_tol, v_list=v_list, w_list=w_list
    )
bary_sched = _
t_sched = time.perf_counter() - t0
print(f"\nWall time: {t_sched:.2f}s")

print("\n" + "=" * 60)
print(f"  [A] Cold start:       {t_cold:.2f}s")
print(f"  [B] Eps schedule:     {t_sched:.2f}s   ({t_cold / t_sched:.2f}x)")
print("=" * 60)

# ── Plot ───────────────────────────────────────────────────────────────────
"""
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

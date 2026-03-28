from firedrake import *
from solvers import HeatEquationSolver
from barycenter import _entropy


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
    v = None
    w = None
    total_iters = []

    for level, (V, epsilon) in enumerate(zip(Vs, epsilons)):
        level_tol = tol if level == len(Vs) - 1 else coarse_tol
        print(
            f"\n--- Level {level + 1}/{len(Vs)}: {V.mesh().num_cells()} cells, eps={epsilon:.4f} ---"
        )

        if v is None:
            # Cold start at coarsest level
            v = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
            w = [HeatEquationSolver(V, dt=epsilon / 2) for _ in range(num_dists)]
            for i in range(num_dists):
                v[i].initialise()
                w[i].initialise()
        else:
            # Rescale to new epsilon, then interpolate onto the finer mesh
            for i in range(num_dists):
                old_epsilon = 2 * float(v[i].dt_const)
                ratio = old_epsilon / epsilon
                v[i].rhs.interpolate(v[i].rhs ** ratio)
                v[i].refine(V, epsilon / 2)
                w[i].refine(V, epsilon / 2)

        # Project input distributions to current mesh level
        curr = [assemble(interpolate(mus[i], V)) for i in range(num_dists)]

        # Sinkhorn loop at this level
        mu = Function(V, name="mu").assign(1.0)
        mu.interpolate(mu / assemble(mu * dx))
        w_prev = Function(V).assign(1.0)
        d = [Function(V).assign(1.0) for _ in range(num_dists)]

        j, res = 0, 1.0
        while res > level_tol and j < maxiter:
            mu.assign(1.0)
            res = 0.0
            for i in range(num_dists):
                w_prev.assign(w[i].rhs)
                v[i].solve()
                w[i].update(curr[i] / v[i].output_function)
                w[i].solve()
                d[i].interpolate(v[i].rhs * w[i].output_function)
                mu.interpolate(mu * (d[i] ** alphas[i]))
                res = max(norm(w_prev - w[i].rhs), res)
            for i in range(num_dists):
                v[i].update(v[i].rhs * (mu / d[i]))
            print(f"  eps={epsilon:.4f}  iter={j:3d}  residual={res:.6e}")
            j += 1

        total_iters.append(j)

    print(f"\nTotal iterations per level: {total_iters}  (sum={sum(total_iters)})")
    return mu, v, w

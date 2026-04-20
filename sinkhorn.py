import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM=True
except:
    print("WARNING: Could not import `tqdm` please run `pip install tqdm` within the virtual environment.")
    print("INFO: Running without progressbar\n")
    TQDM=False

from firedrake import *
from firedrake.pyplot import tripcolor

from solvers import BackwardEuler


# Constants
EPSILONS = [0.1, 0.05, 0.02, 0.01]
MEAN_0 = [0.1, 0.1]
MEAN_1 = [0.5, 0.5]
SIGMA_0 = SIGMA_1 = 0.1


def initialise_env(
    target_size,
    levels,
    mesher,
    make_space
):
    """
    Initialise a function space hierarchy

    Parameters
    ----------
    target_size : The size of the finest grid
    levels      : The number of levels to create
    mesher      : The firedrake mesh maker
    make_space  : The function space maker

    Returns
    -------
    vs : The hierarchy of function spaces
    """
    print("\nInitialising environment (this may take a while)...")
    # Calculate the coarse size to refine (each cell becomes 4)
    start_size = int(target_size / (4**levels))

    mesh = mesher(start_size, start_size)
    hierarchy = MeshHierarchy(mesh, levels)

    vs = []

    for sub_mesh in hierarchy:
        vs.append(make_space(sub_mesh, "CG", 1))

    print("Done!\n")

    return vs

def generate_gaussian(V, mean, sigma):
    """
    Generate simple Gaussian distribution for OT. Handles conversion to Firedrake symbolic function.

    Parameters
    ----------
    V         : Funnction space to define probability density functions on. 
    mean      : 2d array containing mean of mu.
    sigma     : Standard deviation of mu.
    """

    # Set up probability distribution
    mu = Function(V)

    x, y = SpatialCoordinate(V.mesh())
    mu.interpolate((1 / (2 * pi * sigma**2)) * exp(-((x- mean[0])**2 + (y - mean[1])**2) / (2 * sigma**2)))

    # Normalise on mesh
    Imu = assemble(mu*dx)
    mu.assign(mu/Imu)
    return mu

def sinkhorn(
    mu_0,
    mu_1,
    Vs,
    epsilons=[0.5, 0.1, 0.05, 0.01],
    tol=1e-6,
    maxiter=10
):
    """
    Repeat the sinkhorn iteration until the tolerance is reached or maximum iterations.

    Parameters
    ----------
    mu_0     : The source distribution defined on Vs[-1]
    mu_1     : The target distribution defined on Vs[-1]
    Vs       : The hierarchy of function spaces
    epsilons : The regularisation parameter schedule
    tol      : The tolerance at which to stop
    maxiter  : The maximum number of iterations
    """
    print("\nRunning Sinkhorn iteration...")

    phi = Function(Vs[-1])
    psi = Function(Vs[-1])
    curr_mu_0 = Function(Vs[0])
    curr_mu_1 = Function(Vs[0])

    Solver_0 = BackwardEuler(Vs[0], dt=epsilons[0]/2)
    Solver_1 = BackwardEuler(Vs[0], dt=epsilons[0]/2)

    Solver_1.initialise()

    if TQDM:
        iter = tqdm(
            zip(Vs, epsilons),
            total=len(Vs),
            leave=False
        )
    else:
        iter = zip(Vs, epsilons)

    for V, eps in iter:
        if not TQDM: print(f"\nRunning with epsilon={eps}\n")

        Solver_0.refine(V, eps/2)
        Solver_1.refine(V, eps/2)

        curr_mu_0 = assemble(interpolate(mu_0, V))
        curr_mu_1 = assemble(interpolate(mu_1, V))

        i = 0
        res = 1
        while (tol < res) and (i < maxiter):
            Solver_1.solve()
            res = norm(
                Solver_0.function -
                (curr_mu_0 / Solver_1.output_function)
            )
            Solver_0.update(curr_mu_0 / Solver_1.output_function)
            Solver_0.solve()
            Solver_1.update(curr_mu_1 / Solver_0.output_function)

            if TQDM:
                iter.set_description(f"epsilon={eps} residual={res:.6f}")
            else:
                print(res)

            i += 1

    phi.interpolate(epsilons[-1] * ln(Solver_0.function))
    psi.interpolate(epsilons[-1] * ln(Solver_1.function))

    print("Done!\n")
    return phi, psi


if __name__=='__main__':
    # Initialise mesh and function space
    Vs = initialise_env(8192, len(EPSILONS), UnitSquareMesh, FunctionSpace)

    # Generate Gaussian distributions
    mu_0 = generate_gaussian(Vs[-1], MEAN_0, SIGMA_0)
    mu_1 = generate_gaussian(Vs[-1], MEAN_1, SIGMA_1)

    phi, psi = sinkhorn(mu_0, mu_1, Vs, epsilons=EPSILONS)

    print("\nVisualising...")
    Vc = Vs[-1].mesh().coordinates.function_space()
    x, y = SpatialCoordinate(Vs[-1].mesh())
    f = Function(Vc).interpolate(grad(phi))
    Vs[-1].mesh().coordinates.assign(f)

    fig, axes = plt.subplots()
    colors = tripcolor(mu_0, axes=axes)
    fig.colorbar(colors)
    plt.show()

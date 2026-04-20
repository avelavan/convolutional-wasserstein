from abc import ABC, abstractmethod
from firedrake import *


class AbstractHeatEquationSolver(ABC):
    def __init__(self, V, dt=0.1, params=None):
        if params is None:
            params = {"ksp_type": "preonly", "pc_type": "lu"}

        self.V = V
        self.params = params
        self.dt_const = Constant(dt)

        self._allocate()
        self._build_problem()

    def _allocate(self, rhs_value=None):
        """Create trial/test/rhs/output on self.V. If rhs_value given, interpolate into new rhs."""
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.output_function = Function(self.V)
        new_rhs = Function(self.V)
        if rhs_value is not None:
            new_rhs.interpolate(rhs_value)
        self.rhs = new_rhs

    @abstractmethod
    def _build_problem(self):
        """Define self.a, self.L, self.problem, self.solver on current trial/test/output."""

    def solve(self):
        self.solver.solve()
        return self.output_function

    def initialise(self, value=None):
        if value is None:
            self.rhs.assign(1.0)
        else:
            self.rhs.assign(value)

    def update(self, value):
        self.rhs.interpolate(value)

    def update_dt(self, new_dt):
        self.dt_const.assign(new_dt)

    def refine(self, new_V, new_dt):
        old_rhs = self.rhs
        self.V = new_V
        self.dt_const.assign(new_dt)
        self._allocate(rhs_value=old_rhs)
        self._build_problem()


class BackwardEulerSingleStep(AbstractHeatEquationSolver):
    """
    One backward-Euler step of the heat equation:
    (I - dt*Δ)u = u₀, solved by Firedrake.
    """

    def _build_problem(self):
        self.a = (
            self.dt_const * inner(grad(self.u), grad(self.v)) + inner(self.u, self.v)
        ) * dx
        self.L = inner(self.rhs, self.v) * dx
        self.problem = LinearVariationalProblem(self.a, self.L, self.output_function)
        self.solver = LinearVariationalSolver(self.problem, solver_parameters=self.params)


class BackwardEulerMultiStep(BackwardEulerSingleStep):
    """
    Backward-Euler that takes n_steps steps of size dt per solve() call.
    Total time advanced per solve() is n_steps * dt.
    """

    def __init__(self, V, dt=0.1, n_steps=1, params=None):
        self.n_steps = n_steps
        super().__init__(V, dt=dt, params=params)

    def solve(self):
        for _ in range(self.n_steps):
            self.solver.solve()
            self.rhs.assign(self.output_function)
        return self.output_function

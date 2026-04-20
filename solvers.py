from abc import ABC, abstractmethod
from firedrake import *


class AbstractHeatEquationSolver(ABC):

    @abstractmethod
    def solve(self):
        """Run one solve step and return the output function."""

    @abstractmethod
    def initialise(self, value=None):
        """Set the initial RHS value before the first solve."""

    @abstractmethod
    def update(self, value):
        """Update the RHS to a new value between solves."""

    @abstractmethod
    def update_dt(self, new_dt):
        """Change the timestep without rebuilding the solver."""

    @abstractmethod
    def refine(self, new_V, new_dt):
        """Transfer state into a finer function space."""


class BackwardEulerSingleStep(AbstractHeatEquationSolver):
    def __init__(self, V, dt=0.1, params=None):
        """
        Solves one backward-Euler step of the heat equation, yielding the
        modified Helmholtz problem (I - dt*Δ)u = u₀ solved by Firedrake.

        Parameters
        ----------
        V      : The function space the equation is solved in
        dt     : The time step (single step only)
        params : The Firedrake solver parameters
        """
        if params is None:
            params = {"ksp_type": "preonly", "pc_type": "lu"}

        self.params = params
        self.dt_const = Constant(dt)

        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.rhs = Function(V)
        self.output_function = Function(V)

        self._build_problem()

    def _build_problem(self):
        self.a = (
            self.dt_const * inner(grad(self.u), grad(self.v)) + inner(self.u, self.v)
        ) * dx
        self.L = inner(self.rhs, self.v) * dx
        self.problem = LinearVariationalProblem(self.a, self.L, self.output_function)
        self.solver = LinearVariationalSolver(self.problem, solver_parameters=self.params)

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
        self.dt_const.assign(new_dt)

        self.u = TrialFunction(new_V)
        self.v = TestFunction(new_V)
        self.output_function = Function(new_V)
        self.rhs = assemble(interpolate(self.rhs, new_V))

        self._build_problem()

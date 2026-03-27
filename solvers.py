from firedrake import *


class HeatEquationSolver:
    def __init__(self, V, dt=0.1, params=None):
        """
        A heat equation solver that uses a single step of backward euler.
        This results in a modified Helmholtz equation which can be solved by Firedrake.

        Parameters
        ----------
        V      : The function space the heat equation is solved in
        dt     : The time step (only one timestep is completed)
        params : The firedrake solver parameters
        """
        if params is None:
            params = {"ksp_type": "preonly", "pc_type": "lu"}

        self.dt_const = Constant(dt)

        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.rhs = Function(V)
        self.output_function = Function(V)

        self.a = (
            self.dt_const * inner(grad(self.u), grad(self.v)) + inner(self.u, self.v)
        ) * dx
        self.L = inner(self.rhs, self.v) * dx

        self.problem = LinearVariationalProblem(self.a, self.L, self.output_function)
        self.params = params
        self.solver = LinearVariationalSolver(self.problem, solver_parameters=params)

    def solve(self):
        """
        A wrapper for the Firedrake LinearVariationalSolver solve.

        Returns
        -------
        output_function : The resulting solved function
        """
        self.solver.solve()
        return self.output_function

    def initialise(self, value=None):
        """
        Initialise the function to all ones to prevent blow-up
        """
        if value is None:
            self.rhs.assign(1.0)
        else:
            self.rhs.assign(value)

    def update(self, value):
        """
        Set the new value for the initial value function.

        Parameters
        ----------
        value : The value to assign to the value function.
        """
        self.rhs.interpolate(value)

    def update_dt(self, new_dt):
        """
        Update the timestep of the solver.

        Parameters
        ----------
        new_dt : The new timestep
        """
        self.dt_const.assign(new_dt)

    def refine(self, new_V, new_dt):
        """
        Transfer the current potential into a new refined space

        Parameters
        ----------
        new_V  : The new function space
        new_dt : The new timestep
        """
        self.dt_const.assign(new_dt)

        # Create new functions in the space
        self.u = TrialFunction(new_V)
        self.v = TestFunction(new_V)
        self.output_function = Function(new_V)

        # Assign the current function to the new space
        self.rhs = assemble(interpolate(self.rhs, new_V))

        # Setup problem and solver
        self.a = (
            self.dt_const * inner(grad(self.u), grad(self.v)) + inner(self.u, self.v)
        ) * dx
        self.L = inner(self.rhs, self.v) * dx
        self.problem = LinearVariationalProblem(self.a, self.L, self.output_function)
        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=self.params
        )

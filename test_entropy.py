from firedrake import *
from barycenter import _entropy

# Standard 1D grid on [0, 1]
N = 200
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh, "CG", 1)

# Define a Gaussian: mean=0.5, std=0.1
x, = SpatialCoordinate(mesh)
mean = Constant(0.5)
sigma = Constant(0.1)

mu_expr = exp(-0.5 * ((x - mean) / sigma) ** 2)

mu = Function(V).interpolate(mu_expr)

# Normalise so mu integrates to 1
mass = assemble(mu * dx)
mu.interpolate(mu / mass)

entropy = _entropy(mu)
print(f"Entropy of Gaussian (mean=0.5, sigma=0.1): {entropy:.6f}")

# Analytical entropy of Gaussian: 0.5 * ln(2 * pi * e * sigma^2)
import math
sigma_val = 0.1
analytical = 0.5 * math.log(2 * math.pi * math.e * sigma_val**2)
print(f"Analytical entropy:                         {analytical:.6f}")

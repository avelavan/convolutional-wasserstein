import math
import pytest
from firedrake import *
from barycenter import _entropy, _entropic_sharpening


@pytest.fixture(scope="module")
def mesh_and_space():
    N = 200
    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh, "CG", 1)
    return mesh, V


@pytest.fixture(scope="module")
def gaussian_mu(mesh_and_space):
    mesh, V = mesh_and_space
    x, = SpatialCoordinate(mesh)
    expr = exp(-0.5 * ((x - Constant(0.5)) / Constant(0.1)) ** 2)
    mu = Function(V).interpolate(expr)
    mu.interpolate(mu / assemble(mu * dx))
    return mu


@pytest.fixture(scope="module")
def gaussian_mu2(mesh_and_space):
    mesh, V = mesh_and_space
    x, = SpatialCoordinate(mesh)
    expr = exp(-0.5 * ((x - Constant(0.7)) / Constant(0.1)) ** 2)
    mu2 = Function(V).interpolate(expr)
    mu2.interpolate(mu2 / assemble(mu2 * dx))
    return mu2


def test_entropy_gaussian(gaussian_mu):
    sigma_val = 0.1
    analytical = 0.5 * math.log(2 * math.pi * math.e * sigma_val**2)
    computed = _entropy(gaussian_mu)
    assert abs(computed - analytical) < 0.05, (
        f"Entropy {computed:.6f} too far from analytical {analytical:.6f}"
    )


def test_entropic_sharpening_entropy(gaussian_mu, gaussian_mu2):
    h0 = max(_entropy(gaussian_mu), _entropy(gaussian_mu2))
    sharp_mu = _entropic_sharpening(gaussian_mu, h0)
    assert abs(_entropy(sharp_mu) - h0) < 0.05, (
        "Sharpened entropy should match h0"
    )


def test_entropic_sharpening_mass(gaussian_mu, gaussian_mu2):
    h0 = max(_entropy(gaussian_mu), _entropy(gaussian_mu2))
    sharp_mu = _entropic_sharpening(gaussian_mu, h0)
    mass = assemble(sharp_mu * dx)
    assert abs(mass - 1.0) < 1e-4, f"Mass {mass:.6f} should be ~1"

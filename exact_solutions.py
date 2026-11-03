import numpy as np
from enthalpy import Dirichlet
from cattaneo import CattaneoSolver


# Stefan problem on x > 0 with theta(x,0) = -theta_s and theta(0,t) = 1
# the front is s(t) = 2 lam sqrt(t)

from scipy.special import erf, erfc, erfcx
from scipy.optimize import brentq

def neumann_lambda(mat, theta_s):
    nu = 1.0 / np.sqrt(mat.alpha_s)
    St_s = mat.rc * theta_s * mat.St

    def F(lam):
        # exp(-z^2)/erfc(z) = 1/erfcx(z), which does not underflow for large z
        return (lam * np.sqrt(np.pi) - mat.St * np.exp(-lam**2) / erf(lam)
                + St_s / nu / erfcx(nu * lam))

    return brentq(F, 1e-10, 10.0)

def neumann_theta(x, t, mat, theta_s, lam=None):
    lam = neumann_lambda(mat, theta_s) if lam is None else lam
    nu = 1.0 / np.sqrt(mat.alpha_s)
    s = 2.0 * lam * np.sqrt(t)
    th_l = 1.0 - erf(x / (2.0 * np.sqrt(t))) / erf(lam)
    th_s = -theta_s + theta_s * erfc(x / (2.0 * np.sqrt(mat.alpha_s * t))) / erfc(nu * lam)
    return np.where(x < s, th_l, th_s), s

def cell_average(f, N, h, M=64):
    # average of f over each cell [ih, (i+1)h] with M midpoints
    xs = (np.arange(N)[:, None] + (np.arange(M)[None, :] + 0.5) / M) * h
    return f(xs).mean(axis=1)


def tw_H(xi, mat, V, tau):
    # enthalpy of the travelling wave: H = u + Hm in the liquid, H = 0 in the solid
    b = 1.0 - tau * V**2
    return np.where(xi < 0.0, mat.Hm * np.exp(-V * np.minimum(xi, 0.0) / b) / b, 0.0)

def tw_theta(xi, mat, V, tau):
    return np.where(xi < 0.0, tw_H(xi, mat, V, tau) - mat.Hm, 0.0)

def tw_solver(N, mat, V, tau, s0, L=1.0):
    # solver on [0, L] initialized with the wave (q = V H) and the exact temperature at x = 0
    h = L / N
    th_left = lambda t, *X: tw_theta(-s0 - V * t, mat, V, tau)
    H0 = cell_average(lambda X: tw_H(X - s0, mat, V, tau), N, h)
    q0 = [V * tw_H(np.arange(N + 1) * h - s0, mat, V, tau)]
    return CattaneoSolver((N,), h, mat, [(Dirichlet(th_left), Dirichlet(0.0))], H0, tau, q0=q0)

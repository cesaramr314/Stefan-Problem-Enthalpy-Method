import numpy as np
from enthalpy import EnthalpySolver, beta, laplacian


class CattaneoSolver(EnthalpySolver):
    # same arguments as EnthalpySolver, plus the relaxation time tau, the initial flux q0
    # (list with one array per axis; by default q0 = -grad u0) and the viscosity coefficient kappa
    def __init__(self, shape, h, mat, bcs, H0, tau, q0=None, t0=0.0, cfl=0.9, kappa=0.25):
        super().__init__(shape, h, mat, bcs, H0, t0=t0, cfl=1.0)
        self.tau = tau
        self.eps = kappa * h * np.sqrt(mat.lip / tau) if tau > 0 else 0.0

        # largest root of the stability condition; for tau = 0 it is the parabolic bound
        mu, e = 2.0 / self.dt_max, self.eps
        b = 2.0 - 2.0 * e * mu * tau
        self.dt_max = cfl * (b + np.sqrt(b**2 + 16.0 * tau * mu * (1.0 + e))) / (2.0 * mu * (1.0 + e))

        self._grad_u()
        self.q = [-g for g in self.grad] if q0 is None else [np.array(qa, dtype=np.float64) for qa in q0]

    def _grad_u(self):
        # gradient of u on the faces, including the boundary faces
        beta(self.H, self.mat, out=self.U[self.inner], tmp=self.tmp)
        self._fill_ghosts()
        self.grad = []
        for ax in range(self.d):
            idx = list(self.inner); idx[ax] = slice(None)
            self.grad.append(np.diff(self.U[tuple(idx)], axis=ax) / self.h)

    def _boundary_flux(self):
        # incoming heat through the boundary faces
        total = sum(np.take(q, 0, axis=ax).sum() - np.take(q, -1, axis=ax).sum() for ax, q in enumerate(self.q))
        return total * self.h**(self.d - 1)

    def step(self, dt=None):
        dt = self.dt_max if dt is None else min(dt, self.dt_max)
        self._grad_u()
        for ax, q in enumerate(self.q):
            q *= self.tau
            q -= dt * self.grad[ax]
            q /= (self.tau + dt)
            self.H -= (dt / self.h) * np.diff(q, axis=ax)
        self.boundary_energy += dt * self._boundary_flux()

        # numerical viscosity: eps times the step of the Fickian solver
        if self.eps > 0:
            laplacian(self.U, self.h, out=self.lap)
            self.H += (dt * self.eps) * self.lap
            self.boundary_energy += dt * self.eps * EnthalpySolver._boundary_flux(self)
        self.t += dt

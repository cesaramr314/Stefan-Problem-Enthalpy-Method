from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Material:
    St: float               # Stefan number
    rc: float = 1.0         # c_s / c_l
    rk: float = 1.0         # k_s / k_l

    @property
    def Hm(self):           # the phase change happens for H in [0, Hm]
        return 1.0 / self.St

    @property
    def alpha_s(self):      # diffusivity of the solid relative to the liquid
        return self.rk / self.rc

    @property
    def lip(self):          # Lipschitz constant of beta
        return max(1.0, self.alpha_s)


def beta(H, mat, out, tmp):
    # Kirchhoff potential u = alpha_s*min(H,0) + max(H-Hm,0)
    # out and tmp are buffers reused along the iterations (slightly faster than np.where)
    np.minimum(H, 0.0, out=out)
    out *= mat.alpha_s
    np.subtract(H, mat.Hm, out=tmp)
    np.maximum(tmp, 0.0, out=tmp)
    out += tmp
    return out


def theta_from_u(u, mat):
    # temperature from the Kirchhoff potential
    return np.where(u < 0.0, u / mat.rk, u)

def u_from_theta(theta, mat):
    return np.where(theta < 0.0, mat.rk * theta, theta)

def H_from_theta(theta, mat, liquid=None):
    # enthalpy from temperature; at theta = 0 it is multivalued, so we take the value given by the phase
    # (liquid is a boolean array; by default theta = 0 is liquid)
    H = np.where(theta < 0.0, mat.rc * theta, theta + mat.Hm)
    if liquid is not None:
        H = np.where(theta == 0.0, np.where(liquid, mat.Hm, 0.0), H)
    return H

def liquid_fraction(H, mat):
    # f = min(1, max(H/Hm, 0)); the interface is f = 1/2
    return np.clip(H / mat.Hm, 0.0, 1.0)


def laplacian(U, h, out):
    # (2d+1)-point Laplacian, works for d = 1, 2, 3
    # the borders of U are ghost cells; out is a buffer
    d = U.ndim
    inner = (slice(1, -1),) * d
    np.multiply(U[inner], -2.0 * d, out=out)

    for ax in range(d):
        lo = list(inner); lo[ax] = slice(0, -2)
        hi = list(inner); hi[ax] = slice(2, None)
        out += U[tuple(lo)]
        out += U[tuple(hi)]
    out *= 1.0 / h**2
    return out


class Dirichlet:
    # prescribed temperature on the face; value is a number or a function f(t, *x)
    def __init__(self, value):
        self.value = value

class Neumann:
    # incoming heat flux per unit area (0 = insulated face)
    def __init__(self, flux_in=0.0):
        self.flux_in = flux_in


class EnthalpySolver:
    """
    Explicit finite volume scheme for H_t = Laplacian(beta(H)) on a box in d = 1, 2, 3 dimensions.

    Args:
        shape (tuple): Number of cells (N1, ..., Nd).
        h (float): Cell size, the same for every axis.
        mat (Material): Material parameters.
        bcs (list): Pairs (bc_left, bc_right) of Dirichlet/Neumann conditions, one per axis.
        H0 (numpy.ndarray): Initial enthalpy of shape `shape`.
        t0 (float): Initial time. Defaults to 0.
        cfl (float): Fraction of the largest monotone time step. Defaults to 0.9.
    """
    def __init__(self, shape, h, mat, bcs, H0, t0=0.0, cfl=0.9):
        self.d = len(shape)
        self.shape, self.h, self.mat, self.bcs = tuple(shape), h, mat, bcs
        self.t = t0
        self.H = np.array(H0, dtype=np.float64).reshape(self.shape)
        self.U = np.zeros(tuple(n + 2 for n in shape))      # Kirchhoff potential with ghost cells
        self.lap = np.empty(self.shape)
        self.tmp = np.empty(self.shape)
        self.inner = (slice(1, -1),) * self.d
        self.boundary_energy = 0.0                          # heat that entered through the boundary

        # largest diagonal coefficient of the stencil (a Dirichlet face adds 1);
        # dt_max ensures monotonicity
        diag = 2 * self.d + sum(1 for l, r in bcs if isinstance(l, Dirichlet) or isinstance(r, Dirichlet))
        self.dt_max = cfl * h**2 / (mat.lip * diag)

        # coordinates of the centers of the boundary faces
        centers = [(np.arange(n) + 0.5) * h for n in shape]
        self.face_X = {}
        for ax in range(self.d):
            for side, x_b in ((0, 0.0), (1, shape[ax] * h)):
                grids = [c if k != ax else np.array([x_b]) for k, c in enumerate(centers)]
                self.face_X[ax, side] = np.meshgrid(*grids, indexing="ij")

    def _ghost_index(self, ax, side):
        # indices of the ghost layer and of the adjacent inner layer
        g = list(self.inner); a = list(self.inner)
        g[ax] = slice(0, 1) if side == 0 else slice(-1, None)
        a[ax] = slice(1, 2) if side == 0 else slice(-2, -1)
        return tuple(g), tuple(a)

    def _fill_ghosts(self):
        U, h = self.U, self.h
        for ax, pair in enumerate(self.bcs):
            for side, bc in enumerate(pair):
                g, a = self._ghost_index(ax, side)
                if isinstance(bc, Dirichlet):
                    # the value on the face is the mean of the ghost and the inner cell
                    val = bc.value
                    th_b = val(self.t, *self.face_X[ax, side]) if callable(val) else val
                    u_b = u_from_theta(np.asarray(th_b, dtype=float), self.mat)
                    np.subtract(2.0 * u_b, U[a], out=U[g])
                else:
                    np.add(U[a], h * bc.flux_in, out=U[g])

    def _boundary_flux(self):
        # incoming heat per unit time: sum over the faces of h^(d-1) * (U_g - U_a)/h
        U, total = self.U, 0.0
        for ax in range(self.d):
            for side in (0, 1):
                g, a = self._ghost_index(ax, side)
                total += np.sum(U[g] - U[a])
        return total * self.h**(self.d - 2)

    def step(self, dt=None):
        dt = self.dt_max if dt is None else min(dt, self.dt_max)
        beta(self.H, self.mat, out=self.U[self.inner], tmp=self.tmp)
        self._fill_ghosts()
        laplacian(self.U, self.h, out=self.lap)
        self.boundary_energy += dt * self._boundary_flux()
        self.lap *= dt
        self.H += self.lap
        self.t += dt

    def run(self, t_end, callback=None, every=1):
        # the last step is shortened to end exactly at t_end
        n = 0
        while self.t < t_end - 1e-14:
            self.step(t_end - self.t)
            n += 1
            if callback is not None and n % every == 0:
                callback(self)
        return n

    # -- diagnostics -----------------
    @property
    def theta(self):
        u = beta(self.H, self.mat, out=np.empty(self.shape), tmp=self.tmp)
        return theta_from_u(u, self.mat)

    @property
    def fraction(self):
        return liquid_fraction(self.H, self.mat)

    def total_enthalpy(self):
        return self.H.sum() * self.h**self.d

    def liquid_volume(self):
        return self.fraction.sum() * self.h**self.d

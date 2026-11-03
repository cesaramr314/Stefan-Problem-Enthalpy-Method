import numpy as np
from enthalpy import H_from_theta


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from IPython.display import HTML

plt.rcParams["animation.embed_limit"] = 150          # MB, for to_jshtml

WATER = "#3F76E4"           # minecraft water :)
ICE = "#8DB4FE"             # minecraft ice   :)
SNOW = "#EEF5FF"            # light blue, almost white
INK = "#1B2B45"             # text and lines

phase_cmap = LinearSegmentedColormap.from_list("phase", [SNOW, WATER])     # f = 0 solid, f = 1 liquid


def record(solver, t_end, n_frames=60, power=2.0, keep_theta=True):
    """
    Runs the solver up to t_end storing snapshots at the times t0 + (t_end - t0)*(k/n_frames)^power.

    Args:
        solver (EnthalpySolver): Solver at its initial state.
        t_end (float): Final time.
        n_frames (int): Number of frames after the initial one. Defaults to 60.
        power (float): Spacing of the frames in time. Defaults to 2.
        keep_theta (bool): Whether to store the temperature too. Defaults to True.

    Returns:
        tuple: times of shape (n_frames+1,), liquid fractions of shape (n_frames+1, *shape)
            and temperatures of the same shape (None if keep_theta is False).
    """
    t0 = solver.t
    targets = t0 + (t_end - t0) * (np.arange(1, n_frames + 1) / n_frames)**power
    times, frac, theta = [t0], [solver.fraction.astype(np.float32)], []
    if keep_theta:
        theta.append(solver.theta.astype(np.float32))

    for tk in targets:
        solver.run(tk)
        times.append(solver.t)
        frac.append(solver.fraction.astype(np.float32))
        if keep_theta:
            theta.append(solver.theta.astype(np.float32))
    return np.array(times), np.array(frac), (np.array(theta) if keep_theta else None)


def H_from_phase(phi, h, mat, theta_s, theta_l):
    # enthalpy of a solid {phi > 0} at temperature theta_s surrounded by liquid at temperature theta_l,
    # the cells cut by the interface get a mixture of both (anti-aliasing)
    w = np.clip(0.5 + phi / h, 0.0, 1.0)
    return w * H_from_theta(theta_s, mat, liquid=False) + (1 - w) * H_from_theta(theta_l, mat, liquid=True)


def segment_distance(P, A, B):
    # distance from the points P (..., d) to the segment [A, B]
    AB = B - A
    s = np.clip(((P - A) @ AB) / (AB @ AB), 0.0, 1.0)
    return np.linalg.norm(P - A - s[..., None] * AB, axis=-1)


def dendrite_segments(center, directions, side_dirs, length, n_side, shrink):
    # main arms along `directions`; each one has n_side side branches along side_dirs,
    # shorter as they get closer to the tip
    segs = []
    for e, sides in zip(directions, side_dirs):
        segs.append((center, center + length * e))
        for j in range(1, n_side + 1):
            p = center + length * e * j / (n_side + 1)
            l = shrink * length * (1 - j / (n_side + 1))
            segs += [(p, p + l * v) for v in sides]
    return segs


def snowflake_2d(X, Y, center=(0.5, 0.5), length=0.36, width=0.018, n_side=4):
    # signed thickness: positive inside a six-fold dendritic crystal
    c = np.array(center)
    P = np.stack([X, Y], axis=-1)
    angles = np.pi / 2 + np.arange(6) * np.pi / 3
    unit = lambda a: np.array([np.cos(a), np.sin(a)])
    dirs = [unit(a) for a in angles]
    sides = [[unit(a + np.pi / 3), unit(a - np.pi / 3)] for a in angles]
    segs = dendrite_segments(c, dirs, sides, length, n_side, shrink=0.55)
    dist = np.min([segment_distance(P, A, B) for A, B in segs], axis=0)
    r = np.linalg.norm(P - c, axis=-1)
    return width * (1.0 + 1.2 * np.exp(-(r / 0.08)**2)) - dist          # thicker core


def dendrite_3d(X, Y, Z, center=(0.5, 0.5, 0.5), length=0.42, width=0.027, n_side=3):
    # cubic dendrite: six arms along the axes with side branches orthogonal to them
    c = np.array(center)
    P = np.stack([X, Y, Z], axis=-1)
    E = np.eye(3)
    dirs, sides = [], []
    for i in range(3):
        for sgn in (1, -1):
            dirs.append(sgn * E[i])
            sides.append([s * E[j] for j in range(3) if j != i for s in (1, -1)])
    segs = dendrite_segments(c, dirs, sides, length, n_side, shrink=0.6)
    dist = np.min([segment_distance(P, A, B) for A, B in segs], axis=0)
    r = np.linalg.norm(P - c, axis=-1)
    return width * (1.0 + 0.8 * np.exp(-(r / 0.08)**2)) - dist


def animate_1d(x, times, F, TH, interval=80):
    # top: temperature over the phases; bottom: space-time diagram of the phases
    L = x[-1] + (x[1] - x[0]) / 2
    th_min, th_max = TH.min(), TH.max()
    pad = 0.08 * (th_max - th_min)
    fig, (ax, ax_st) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw=dict(height_ratios=[1.3, 1]))

    phase = ax.imshow(F[0][None, :], extent=[0, L, th_min - pad, th_max + pad], aspect="auto",
                      cmap=phase_cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.axhline(0.0, color="white", lw=1, ls="--", alpha=0.8)
    line, = ax.plot(x, TH[0], color=INK, lw=2)
    ax.set_xlim(0, L); ax.set_ylim(th_min - pad, th_max + pad)
    ax.set_xticks([]); ax.set_ylabel(r"$\theta(x,t)$")
    title = ax.set_title(f"t = {times[0]:.4f}", loc="left", color=INK, family="monospace")

    # pcolormesh instead of imshow, since the frames are not uniform in time
    ST = np.full((len(times), len(x)), np.nan)
    t_edges = np.concatenate([[times[0]], (times[1:] + times[:-1]) / 2, [times[-1]]])
    x_edges = np.linspace(0, L, len(x) + 1)
    mesh = ax_st.pcolormesh(x_edges, t_edges, np.ma.masked_invalid(ST), cmap=phase_cmap, vmin=0, vmax=1)
    ax_st.set_facecolor("#F4F6FA")
    ax_st.set_xlabel("x"); ax_st.set_ylabel("t")
    for a in (ax, ax_st):
        for sp in a.spines.values():
            sp.set_color("#B8C4D6")
    fig.tight_layout()

    def update(k):
        phase.set_data(F[k][None, :])
        line.set_ydata(TH[k])
        ST[:k + 1] = F[:k + 1]
        ST[k + 1:] = np.nan
        mesh.set_array(np.ma.masked_invalid(ST).ravel())
        title.set_text(f"t = {times[k]:.4f}")
        return phase, line, mesh, title

    plt.close(fig)
    return FuncAnimation(fig, update, frames=len(times), interval=interval)


def animate_2d(times, F, TH, L=1.0, interval=70, isotherms=True):
    # liquid fraction with the interface f = 1/2 and (optionally) some isotherms
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    im = ax.imshow(F[0].T, origin="lower", extent=[0, L, 0, L], cmap=phase_cmap, vmin=0, vmax=1,
                   interpolation="bilinear")
    n = F.shape[1]
    xc = (np.arange(n) + 0.5) * L / n
    levels = np.linspace(TH.min(), TH.max(), 14)[1:-1]
    levels = levels[np.abs(levels) > 0.05 * (TH.max() - TH.min())]       # theta = 0 is the interface itself
    iso_colors = ["#7FA3E0" if l < 0 else "white" for l in levels]
    label = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left", color="white",
                    fontsize=12, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc=INK, ec="none", alpha=0.35))
    contours = []

    def update(k):
        for c in contours:
            c.remove()
        contours.clear()
        im.set_data(F[k].T)
        if isotherms:
            contours.append(ax.contour(xc, xc, TH[k].T, levels=levels, colors=iso_colors, linewidths=0.6, alpha=0.4))
        contours.append(ax.contour(xc, xc, F[k].T, levels=[0.5], colors=[INK], linewidths=0.9, alpha=0.55))
        label.set_text(f"t = {times[k]:.4f}")
        return [im, label]

    update(0)
    plt.close(fig)
    return FuncAnimation(fig, update, frames=len(times), interval=interval)


def ice_mesh(f, h, level=0.5):
    # isosurface of the solid {f < 1/2}; padding with liquid closes the surface at the walls,
    # and the small blur removes the terraces of the clipped fraction
    fp = gaussian_filter(np.pad(f, 1, constant_values=1.0), 0.8)
    verts, faces, _, _ = marching_cubes(fp, level=level)
    return (verts - 0.5) * h, faces                  # cell centers are at (i + 1/2) h


def shade(verts, faces, base, light=(-0.4, -0.6, 0.7), ambient=0.62):
    # flat (Lambert + specular) shading of each triangle
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    l = np.array(light) / np.linalg.norm(light)
    lam = np.abs(n @ l)
    rgb = np.array(to_rgb(base))[None, :] * (ambient + (1 - ambient) * lam[:, None]) + 0.35 * lam[:, None]**24
    return np.clip(rgb, 0, 1)


def draw_box(ax, L):
    # thin edges and a faint floor; the liquid itself is transparent
    r = [0, L]
    for s in r:
        for t in r:
            ax.plot(r, [s, s], [t, t], color=WATER, lw=0.8, alpha=0.5)
            ax.plot([s, s], r, [t, t], color=WATER, lw=0.8, alpha=0.5)
            ax.plot([s, s], [t, t], r, color=WATER, lw=0.8, alpha=0.5)
    floor = [[(0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0)]]
    ax.add_collection3d(Poly3DCollection(floor, facecolor=WATER, alpha=0.08, edgecolor="none"))


def animate_3d(times, F, h, interval=90, azim0=30, spin=0.6, elev=22):
    # ice as a shaded isosurface, with the camera turning slowly
    L = F.shape[1] * h
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.08)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    draw_box(ax, L)
    label = fig.text(0.04, 0.95, "", color=INK, fontsize=12, family="monospace")
    surf = []

    def update(k):
        for s in surf:
            s.remove()
        surf.clear()
        if F[k].min() < 0.5:
            verts, faces = ice_mesh(F[k], h)
            poly = Poly3DCollection(verts[faces], facecolors=shade(verts, faces, ICE), edgecolor="none", alpha=0.97)
            ax.add_collection3d(poly)
            surf.append(poly)
        ax.view_init(elev=elev, azim=azim0 + spin * k)
        label.set_text(f"t = {times[k]:.4f}")
        return surf + [label]

    update(0)
    plt.close(fig)
    return FuncAnimation(fig, update, frames=len(times), interval=interval)


def phase_panels(ax, ax_st, x, times, F, TH, TH_exact=None):
    # temperature over the phases (exact solution dashed) and space-time diagram of the phases;
    # returns the function that draws the frame k
    L = x[-1] + (x[1] - x[0]) / 2
    th_min, th_max = TH.min(), TH.max()
    pad = 0.08 * (th_max - th_min)

    phase = ax.imshow(F[0][None, :], extent=[0, L, th_min - pad, th_max + pad], aspect="auto",
                      cmap=phase_cmap, vmin=0, vmax=1, interpolation="bilinear")
    ax.axhline(0.0, color="white", lw=1, ls="--", alpha=0.8)
    if TH_exact is not None:
        exact, = ax.plot(x, TH_exact[0], color="white", lw=1.6, ls=(0, (4, 3)))
    line, = ax.plot(x, TH[0], color=INK, lw=2)
    ax.set_xlim(0, L); ax.set_ylim(th_min - pad, th_max + pad)
    ax.set_xticks([]); ax.set_ylabel(r"$\theta(x,t)$")

    # pcolormesh instead of imshow, since the frames may not be uniform in time
    ST = np.full((len(times), len(x)), np.nan)
    t_edges = np.concatenate([[times[0]], (times[1:] + times[:-1]) / 2, [times[-1]]])
    x_edges = np.linspace(0, L, len(x) + 1)
    mesh = ax_st.pcolormesh(x_edges, t_edges, np.ma.masked_invalid(ST), cmap=phase_cmap, vmin=0, vmax=1)
    ax_st.set_facecolor("#F4F6FA")
    ax_st.set_xlabel("x"); ax_st.set_ylabel("t")
    for a in (ax, ax_st):
        for sp in a.spines.values():
            sp.set_color("#B8C4D6")

    def update(k):
        phase.set_data(F[k][None, :])
        line.set_ydata(TH[k])
        if TH_exact is not None:
            exact.set_ydata(TH_exact[k])
        ST[:k + 1] = F[:k + 1]
        ST[k + 1:] = np.nan
        mesh.set_array(np.ma.masked_invalid(ST).ravel())

    return update


def animate_1d_taus(x, runs, interval=80):
    # one row per tau (temperature | space-time diagram) with a common clock;
    # runs[tau] = (times, F, TH, TH_exact), with the same times for every tau
    taus = list(runs)
    times = runs[taus[0]][0]
    fig, axs = plt.subplots(len(taus), 2, figsize=(11, 2.4 * len(taus)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.6, 1]))
    updates = [phase_panels(ax, ax_st, x, *runs[tau]) for (ax, ax_st), tau in zip(axs, taus)]
    for (ax, _), tau in zip(axs, taus):
        ax.set_title(rf"$\tau = {tau}$", loc="left", color=INK)
    title = fig.suptitle(f"t = {times[0]:.4f}", x=0.02, ha="left", color=INK, family="monospace")
    fig.tight_layout()

    def frame(k):
        for update in updates:
            update(k)
        title.set_text(f"t = {times[k]:.4f}")

    plt.close(fig)
    return FuncAnimation(fig, frame, frames=len(times), interval=interval)

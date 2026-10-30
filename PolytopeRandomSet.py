import numpy as np
import matplotlib.pyplot as plt
from Polytope import Polytope
from utils import UNIT_VECTORS, ALPHAS, capacity_level_bounds, refine_level_by_lp, make_pairplot_axes


class PolytopeRandomSet:
    """Random set whose realizations are polytopes, represented by a finite sample.

    Parameters
    ----------
    sample : list of Polytope
        Observed realizations X_1, ..., X_n of the random set.

    Attributes
    ----------
    sample : list of Polytope
    n : int
        Sample size.
    """

    def __init__(self, sample):
        self._supp_funcs = None
        self.sample = []
        self.n = None

        self.set_sample(sample)


    def set_sample(self, sample):
        """Replace the sample and recompute the cached support functions.

        Parameters
        ----------
        sample : list of Polytope
        """
        self.sample = sample
        self.n = len(sample)
        self.compute_supp_funcs()

    def compute_supp_funcs(self):
        """Compute and cache the support functions of all sample sets.

        Sets `self._supp_funcs`, an array of shape (360, n): entry (i, j) is the
        support function of the j-th set in the i-th direction of UNIT_VECTORS.
        """
        self._supp_funcs = np.array([s.support_function(UNIT_VECTORS) for s in self.sample], dtype=np.float64).T

    def quantile(self, alpha):
        """Quantile set K_alpha of the random set.

        Parameters
        ----------
        alpha : float
            Level in [0, 1].

        Returns
        -------
        Polytope
            Polygon ⋂_u {x : <u, x> <= q(u)} with q(u) the (1 - alpha)-quantile of
            the sample support functions. It may be empty.
        """
        return Polytope(A=UNIT_VECTORS, b=np.quantile(self._supp_funcs, 1 - alpha, axis=1))


    def quantiles(self, alphas=ALPHAS):
        """Quantile sets for several levels at once.

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1].

        Returns
        -------
        dict
            Maps str(alpha) to the Polytope K_alpha, in the order of `alphas`.
        """
        quantile_funcs = np.quantile(self._supp_funcs, 1-alphas, axis=1)
        quantiles = np.array([Polytope(A=UNIT_VECTORS, b=q) for q in quantile_funcs])

        return {str(a): q for a,q in zip(alphas,quantiles)}

    def containment_functional(self, polytope):
        """Empirical containment functional P(X is a subset of `polytope`).

       Parameters
       ----------
       polytope : Polytope

       Returns
       -------
       float
           Fraction of sample sets contained in `polytope`.
       """
        return np.mean([polytope.contains(X) for X in self.sample])

    def quantile_containments(self, alphas=ALPHAS):
        """Containment functional of the quantiles: P(X is a subset of K_alpha).

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1].

        Returns
        -------
        (m,) array
            Value of the containment functional for each level.
        """
        return np.array([self.containment_functional(self.quantile(a)) for a in alphas])


    def quantile_capacities(self, alphas=ALPHAS, chunk=20000):
        """Capacity functional of the quantiles: P(X intersects K_alpha).

        For each sample set the largest level k such that it intersects K_k is
        found. Cheap vertex-based lower and upper bounds are computed first,
        and the exact level is determined by linear programs for the sets where
        the bounds differ.

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1], sorted in ascending order.
        chunk : int
            Approximate number of vertices processed at once in the bound
            computation (memory control).

        Returns
        -------
        (m,) array
            Values of capacity functional for each level.

        Raises
        ------
        RuntimeError
            If a linear program fails for a reason other than infeasibility.
        """

        assert np.all(np.diff(alphas) >= 0)
        quantiles = list(self.quantiles(alphas).values())
        m = len(quantiles)

        m_eff = next((k for k, q in enumerate(quantiles) if q.is_empty), m)
        if m_eff == 0:
            return np.zeros(m)

        A = np.asarray(quantiles[0].A, dtype=float)
        B = np.array([np.ravel(q.b) for q in quantiles[:m_eff]])
        B = np.minimum.accumulate(B, axis=0)

        verts = [p.V for p in self.sample]
        starts = np.concatenate(([0], np.cumsum([len(v) for v in verts])))

        lower, upper = capacity_level_bounds(np.vstack(verts), starts, A, B, chunk)
        level = lower.copy()
        for j in np.flatnonzero(lower < upper):
            level[j] = refine_level_by_lp(verts[j], A, B, lower[j], upper[j])

        counts = np.bincount(level, minlength=m + 1)
        return (counts[::-1].cumsum()[::-1])[1:] / self.n

    def bag_quantile(self, dec_places=5):
        """Quantile used as the "bag": the smallest quantile K_alpha that
        contains at least half of the sample sets.

        The largest such alpha is found by bisection on the grid
        {0, 10**(-dec_places), ..., 1}.

        Parameters
        ----------
        dec_places : int
            Number of decimal places of the returned alpha.

        Returns
        -------
        alpha : float
            Level of the bag.
        bag : Polytope
            The quantile K_alpha.
        """
        if self.containment_functional(self.quantile(1)) >= 0.5:
            return float(1), self.quantile(1)
        l,r = 0, 10**dec_places
        while r>l+1:
            m = (l+r)//2
            value = self.containment_functional(self.quantile(m/10**dec_places))
            if value < 0.5:
                r = m
            else:
                l = m
        return l/10**dec_places, self.quantile(l/10**dec_places)


    def median_set(self, dec_places=4):
        """Median set: the last non-empty quantile K_alpha.

        The largest alpha with a non-empty quantile is found by bisection on the
        grid {0, 10**(-dec_places), ..., 1}.

        Parameters
        ----------
        dec_places : int
            Number of decimal places of the returned alpha.

        Returns
        -------
        alpha_max : float
            Level of the median set.
        median : Polytope
            The quantile K_alpha_max.
        """
        if not self.quantile(1).is_empty:
            return float(1), self.quantile(1)
        l,r = 0, 10**dec_places
        while r>l+1:
            m = (l+r)//2
            q = self.quantile(m/10**dec_places)
            if q.chebyshev_radius() < 0:
                r = m
            else:
                l = m
        return l/10**dec_places, self.quantile(l/10**dec_places)


    #############
    ##  PLOTS  ##
    #############

    def plot_quantiles(self, alphas=ALPHAS, ax=None):
        """Plot the quantile contours, colored by level alpha.

        Parameters
        ----------
        alphas : (m,) array
            Levels to plot.
        ax : matplotlib Axes, optional
            Target axis; the current axis is used if None.

        Returns
        -------
        matplotlib Axes
            The axis with equal aspect ratio.
        """
        quantiles_dict = self.quantiles(alphas)

        if ax is None:
            ax = plt.gca()
        for alpha, quantile in quantiles_dict.items():
            if quantile.n_vertices == 0:
                continue

            quantile.minimize_V_rep(prune_V=True)
            quantile.plot2d(ax=ax, patch_args={"edgecolor": plt.get_cmap("gist_rainbow_r")(float(alpha)),
                                               "facecolor": None, "lw": 1.5, "zorder": float(alpha)})
        plt.setp(ax, aspect="equal")
        return ax

    def plot_quantiles_3d(self, alphas=ALPHAS, ax=None):
        """Plot the quantiles in 3D, each placed at height alpha.

        Parameters
        ----------
        alphas : (m,) array
            Levels to plot.
        ax : mpl_toolkits.mplot3d Axes3D, optional
            Target axis; a new 3D axis is created if None.

        Returns
        -------
        Axes3D
        """
        quantiles_dict = self.quantiles(alphas)

        if ax is None:
            ax = plt.subplot(projection='3d', computed_zorder=False)
        for alpha, quantile in quantiles_dict.items():
            if quantile.n_vertices == 0:
                continue

            new_poly = Polytope(V=np.hstack([quantile.V, (float(alpha) * np.ones(quantile.n_vertices)).reshape(-1, 1)]))
            new_poly.plot3d(ax=ax, patch_args={"edgecolor": plt.get_cmap("gist_rainbow_r")(float(alpha)),
                                               "facecolor": None, "lw": 2, "zorder": float(alpha)})
        ax.grid(visible=False)
        ax.set_zlim(0, 1)
        ax.set_zlabel("$\\alpha$", fontsize=12)
        plt.setp(ax, aspect="equalxy")
        ax.set_box_aspect(aspect=None, zoom=0.8)
        return ax

    def bagplot(self, inflation_factor=3, ax=None):
        """Bagplot of the random set (planar case, d = 2).

        Drawn elements: the bag (see `bag_quantile`), the median set (see
        `median_set`) with its centroid marked by a cross, the "loop" (convex
        hull of the bag and all sets inside the fence), and the outlying sets
        in red. The fence is the bag inflated by `inflation_factor` around the
        centroid of the median set; sets not contained in the fence are outliers.

        Parameters
        ----------
        inflation_factor : float
            Factor by which the bag is scaled about the center to get the fence.
        ax : matplotlib Axes, optional
            Target axis; a new axis is created if None.

        Returns
        -------
        matplotlib Axes
        """
        if ax is None:
            ax = plt.subplot()

        bag_alpha, bag = self.bag_quantile()
        med_alpha, med = self.median_set()
        center = med.interior_point("centroid")
        fence = bag.minus(center).affine_map(inflation_factor * np.identity(2)).plus(center)
        fence.minimize_V_rep(prune_V=True)

        outliers = []
        V = []
        for poly in self.sample:
            if fence.contains(poly):
                V.extend(poly.V)
            else:
                outliers.append(poly)
        if len(V) == 0:
            V = np.empty((0, 2))

        for poly in outliers:
            poly.plot(ax=ax, patch_args={"fc": "tomato", "ec": "red", "alpha": 0.5, "zorder": 3},
                   vertex_args={"color": "red", "s": 0, "alpha": 0.5, "zorder": 1})

        loop = Polytope(V=np.concatenate([V, bag.V]))
        loop.minimize_V_rep(prune_V=True)
        loop.plot(ax, patch_args={"ec": "black", "fc": "gainsboro", "fill": True, "ls": (0, (3, 3)), "alpha": 0.6,
                                  "zorder": 3})

        bag.plot(ax, patch_args={"fc": "lightskyblue", "ec": "steelblue", "alpha": 0.8, "zorder": 4})
        med.plot(ax, patch_args={"fc": "steelblue", "ec": "steelblue", "alpha": 0.8, "zorder": 4})
        ax.scatter(center[0], center[1], c="royalblue", marker="x", alpha=0.9, zorder=5)

        return ax

    def pairplot(self, n_of_quantiles=51):
        """Pairplot between level alpha, the containment probability and the volume of K_alpha.

        Parameters
        ----------
        n_of_quantiles : int
            Number of equally spaced levels in [0, 1] to compute.

        Returns
        -------
        fig : matplotlib Figure
        ax : (2, 2) array of matplotlib Axes
        """
        alphas = np.linspace(0,1, n_of_quantiles)
        quantiles = self.quantiles(alphas).values()
        containments = np.array([self.containment_functional(q) for q in quantiles])
        volumes = np.array([q.volume() if q.is_full_dimensional else 0 for q in quantiles])
        n_of_nonempty = np.count_nonzero([not q.is_empty for q in quantiles])

        fig, ax = make_pairplot_axes()

        for i,(x,y) in enumerate(zip([alphas,alphas],[containments, volumes])):
            ax[i,0].plot(x, y, "--", c="black", lw=1, alpha=0.5, zorder=0)
            ax[i,0].scatter(x, y, ec="white", fc="white", lw=2, zorder=1)
            l1 = ax[i,0].scatter(x[:n_of_nonempty], y[:n_of_nonempty], fc="grey", marker="o", lw=0, zorder=2)
            l2 = ax[i,0].scatter(x[n_of_nonempty:], y[n_of_nonempty:], fc="white", ec="gray",s=20, lw=1, zorder=2)
        ax[1,1].plot(containments, volumes, "--", c="black", lw=1, alpha=0.5, zorder=0)
        ax[1,1].scatter(containments[::-1], volumes[::-1], c=alphas[::-1], cmap="gist_rainbow_r", lw=0, zorder=1)

        alpha_max,_ = self.median_set(dec_places=3)
        ax[1,0].axvline(alpha_max, linestyle="--", color="olive", lw=1, alpha=0.5, zorder=0.5)
        ax[1,0].text(alpha_max, 0.95*ax[1,0].get_ylim()[1], "$\\alpha_{\\max}$", c="olive", fontsize=14, ha="right", va="top", usetex=True)

        ax[0,1].legend([l1,l2],["non-empty $K_\\alpha$", "empty $K_\\alpha$"], fontsize=10, loc="lower center")

        return fig, ax



    def symmetry_plot(self, n_of_quantiles=41, seed=None):
        """Plot the halfspace depth of a reference point w.r.t. the quantiles.

        The reference point R is the barycenter of the median set. For each
        level alpha in [0, alpha_max) the depth of R in K_alpha (uniform
        distribution on K_alpha) is plotted. Values near 0.5 indicate that R
        is close to the center of symmetry of the quantile.

        Parameters
        ----------
        n_of_quantiles : int
            Number of equally spaced levels in [0, alpha_max) to compute.
        seed : int, optional
            Seed of the Monte Carlo sampling used for the depth.

        Returns
        -------
        matplotlib Axes
        """
        alpha_max, median_set = self.median_set(dec_places=3)
        R = median_set.interior_point("centroid")
        alphas = np.linspace(0, alpha_max, num=n_of_quantiles, endpoint=False)

        ax = plt.subplot()
        ax.set_xlabel("${\\alpha}$", fontsize=12, usetex=True)
        ax.set_ylabel("$HD(R(\\alpha_{\\max}), Q_\\alpha)$", fontsize=12, usetex=True)
        ax.hlines([0,0.5,1], -1, 2, linestyle="--", color="black", lw=1, alpha=0.5)

        depths = [self.quantile(alpha).halfspace_depth(R, seed=seed) for alpha in alphas]
        plt.setp(ax, xlim=(0, alpha_max), ylim=(-0.025, max(0.525, max(depths)+0.025)))

        ax.plot(alphas, depths, c="black")
        return ax

    def containment_capacity_plot(self, n_of_quantiles=41):
        """Plot the containment and capacity functionals of the quantiles
        against the level alpha.

        Parameters
        ----------
        n_of_quantiles : int
            Number of equally spaced levels in [0, 1] to compute.

        Returns
        -------
        matplotlib Axes
        """
        alphas = np.linspace(0, 1, n_of_quantiles)
        containments = self.quantile_containments(alphas=alphas)
        capacities = self.quantile_capacities(alphas=alphas)

        ax = plt.subplot()
        plt.setp(ax, xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
        ax.hlines([0, 1], -1, 2, color="black", lw=1, alpha=0.2)

        ax.plot(alphas, containments, "--", c="darkblue", lw=1, alpha=0.5, zorder=0)
        l1 = ax.scatter(alphas, containments, fc="dodgerblue", marker="o", lw=0, zorder=2)
        ax.plot(alphas, capacities, "--", c="darkred", lw=1, alpha=0.5, zorder=0)
        l2 = ax.scatter(alphas, capacities, fc="red", marker="o", lw=0, zorder=2)
        ax.legend([l1, l2], ["$P\\,(X \\subseteq K_\\alpha)$", "$P\\,(X \\cap K_\\alpha \\neq \\emptyset)$"],
                  loc="best", bbox_to_anchor=(0.01, 0.04, 0.98, 0.92))
        ax.set_xlabel("$\\alpha$", fontsize=15, usetex=True)
        ax.set_ylabel("Probability", fontsize=12)

        return ax

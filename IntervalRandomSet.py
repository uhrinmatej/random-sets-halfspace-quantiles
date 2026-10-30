"""Interval-valued random sets (d = 1) and their quantiles."""

import numpy as np
import matplotlib.pyplot as plt

from utils import ALPHAS

class Interval:
    """Closed interval [a, b] on the real line.

    Parameters
    ----------
    a, b : float
        Left and right endpoint, a <= b.

    Raises
    ------
    ValueError
        If a > b.
    """

    def __init__(self, a,b):
        if a > b:
            raise ValueError(f"Interval requires a <= b, got a={a}, b={b}")
        self.a = a
        self.b = b

    def support_function(self, direction):
        """Support function in direction u.

        Parameters
        ----------
        direction : float
            Direction u (in 1D typically -1 or +1).

        Returns
        -------
        float
            Support function value.
        """
        return max(direction*self.a, direction*self.b)


class IntervalRandomSet:
    """Random interval, represented by a finite sample of intervals.

    Parameters
    ----------
    sample : list of Interval
        Observed realizations X_1, ..., X_n of the random interval (n >= 1).

    Attributes
    ----------
    sample : list of Interval
    n : int
        Sample size.
    """
    def __init__(self, sample):
        self.sample = []
        self.n = None
        self._supp_funcs = None
        self._a_sorted = None
        self._b_sorted = None

        self.set_sample(sample)

    def set_sample(self, sample):
        """Replace the sample and recompute the cahed support functions.

        Parameters
        ----------
        sample : list of Interval
            Non-empty list.
        """
        self.sample = sample
        self.n = len(sample)
        _a = np.array([s.a for s in self.sample], dtype=float)
        _b = np.array([s.b for s in self.sample], dtype=float)
        self._supp_funcs = np.array([-_a, _b])
        self._a_sorted = np.sort(_a)
        self._b_sorted = np.sort(_b)

    def coverage_function(self, x):
        """Empirical coverage function P(x in X); x can be a scalar or an array.

        Parameters
        ----------
        x : float or array
            Point(s) at which the coverage is evaluated.

        Returns
        -------
        float or array
            Fraction of sample intervals containing x, with the shape of x.
        """
        x = np.asarray(x, dtype=float)
        count = (np.searchsorted(self._a_sorted, x, side="right")
                 - np.searchsorted(self._b_sorted, x, side="left"))
        return count / self.n

    def coverage_quantiles(self, alphas=ALPHAS):
        """Coverage-based quantiles: {x : coverage(x) >= alpha}.

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1].

        Returns
        -------
        list of list of Interval
            One entry per level: the connected components of the level set, in
            increasing order (an empty list if the level set is empty). A component
            can be a single point (a degenerate Interval with a == b).
        """
        p = np.unique(np.concatenate([self._a_sorted, self._b_sorted]))

        n_a = np.searchsorted(self._a_sorted, p, side="right")
        n_b_le = np.searchsorted(self._b_sorted, p, side="right")
        n_b_lt = np.searchsorted(self._b_sorted, p, side="left")

        cov = np.empty(2 * len(p) - 1)
        cov[0::2] = (n_a - n_b_lt) / self.n  # at the endpoints
        cov[1::2] = ((n_a - n_b_le) / self.n)[:-1]  # on the open gaps

        quantiles = []
        for a in alphas:
            inside = np.concatenate(([False], cov >= a - 1e-12, [False]))
            starts = np.flatnonzero(inside[1:-1] & ~inside[:-2])
            ends = np.flatnonzero(inside[1:-1] & ~inside[2:])
            quantiles.append([Interval(p[s // 2], p[e // 2]) for s, e in zip(starts, ends)])
        return quantiles

    def quantiles(self, alphas=ALPHAS):
        """Halfspace-based quantiles K_alpha = [-l, r].

        Here l and r are the (1 - alpha)-quantiles of the sample support
        functions in the directions -1 and +1, i.e. of -a_i and b_i. The
        quantile is empty if r < -l.

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1].

        Returns
        -------
        list of (Interval or None)
            One Interval quantile per level; None if the level set is empty.
        """
        quantile_funcs = np.array([np.quantile(self._supp_funcs, 1-a, axis=1) for a in alphas])
        return [Interval(-l,r) if r>=-l else None for l,r in quantile_funcs]


    def plot_quantiles(self, alphas=ALPHAS, limits=None):
        """Compare halfspace-based and coverage-based quantiles.

        Each non-empty quantile is drawn as a horizontal segment at height
        alpha: blue for halfspace-based, red for coverage-based quantiles.

        Parameters
        ----------
        alphas : (m,) array
            Levels in [0, 1].
        limits : tuple of float, optional
            x-axis limits; defaults to extrema of the sample-

        Returns
        -------
        matplotlib Axes
        """
        if limits is None:
            limits = (self._a_sorted.min(), self._b_sorted.max())

        halfspace = [[q] if q is not None else [] for q in self.quantiles(alphas)]
        coverage = self.coverage_quantiles(alphas)

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))
        for components, color, label in ((halfspace, "blue", "Halfspace-based quantiles"),
                                         (coverage, "red", "Coverage-based quantiles")):
            ax.hlines(y=[a for a, cs in zip(alphas, components) for _ in cs],
                      xmin=[q.a for cs in components for q in cs],
                      xmax=[q.b for cs in components for q in cs],
                      color=color, alpha=0.5, lw=2, snap=False, label=label)

        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("$\\boldsymbol{\\alpha}$", fontsize=15, usetex=True)
        plt.setp(ax, ylim=(-0.02, 1), xlim=limits)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return ax
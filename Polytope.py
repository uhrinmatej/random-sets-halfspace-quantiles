import numpy as np
from pycvxset import Polytope as BasePolytope
from depth.model.DepthEucl import DepthEucl as Depth
from scipy.spatial import Delaunay
from scipy.optimize import linprog


class Polytope(BasePolytope):
    """pycvxset Polytope extended with a support function, uniform sampling and halfspace (Tukey) depth."""

    def support_function(self, unit_vectors):
        """Support function of the polytope, evaluated for each direction in unit_vectors.

        Parameters
        ----------
        unit_vectors : (m, d) array
            Directions u, one per row.

        Returns
        -------
        (m,) array
            Values h(u). The maximum of a linear function over a polytope is
            attained at a vertex, so it is taken over the vertices V only.
        """
        return np.max(np.dot(self.V, unit_vectors.T), axis=0)

    def chebyshev_radius(self):
        """Chebyshev radius of the polytope.

        Overrides the pycvxset `Polytope.chebyshev_radius` method using the HiGHS solver,
        Solves the linear program  max r  s.t.  a_i x + r ||a_i|| <= b_i  for all i,
        where (x, r) are the center and radius of the ball.

        Returns
        -------
        float
            The Chebyshev radius, or -inf if the LP has no solution (e.g. the
            polytope is empty or unbounded). A value of 0 means the polytope has
            no interior (it is lower-dimensional).
        """
        c = np.zeros(self.dim + 1)
        c[-1] = -1  # maximize r
        norms = np.linalg.norm(self.A, axis=1)  # ||a_i||
        res = linprog(c=c, A_ub=np.c_[self.A, norms], b_ub=self.b,
                      bounds=(None, None), method="highs")

        if not res.success:
            return -np.inf

        return res.x[-1]


    def uniform_sampling(self, n, seed=None):
        """Draw n points uniformly from a full-dimensional polytope.

        The polytope is triangulated (Delaunay) into simplices. A simplex is
        chosen with probability proportional to its volume, and a point is then
        drawn uniformly inside it using Dirichlet(1, ..., 1) barycentric weights.

        Returns
        -------
        (n, d) array
            sample of n d-dimensional points
        """
        rng = np.random.default_rng(seed)
        V = np.asarray(self.V, float)
        tri = Delaunay(V)
        S = V[tri.simplices]
        vol = np.abs(np.linalg.det(S[:, 1:, :] - S[:, [0], :]))
        idx = rng.choice(len(S), size=n, p=vol / vol.sum())
        w = rng.dirichlet(np.ones(self.dim + 1), size=n)
        return np.einsum('ni,nij->nj', w, S[idx])

    def halfspace_depth(self, point, n=10_000, seed=None):
        """Halfspace (Tukey) depth of a single point w.r.t. the uniform distribution on the polytope.

        Degenerate cases are handled exactly: a singleton gives 1 if the point
        equals it and 0 otherwise, and a lower-dimensional polytope gives 0.
        Otherwise, the depth is an estimate based on `n` uniform
        sample points, depending on `seed`.

        Parameters
        ----------
        point : (d,) array
        seed : int, optional
            Seed for the random sample.
        n : int
            Sample size of the random sample.

        Returns
        -------
        float
            Halfspace depth of the point.
        """
        point = np.asarray(point, float)
        if self.is_singleton:
            return 1 if np.allclose(point, self.V[0], atol=1e-9) else 0
        if not self.is_full_dimensional:
            return 0

        sample = self.uniform_sampling(n=n, seed=seed)
        return Depth().load_dataset(sample).halfspace(point)[0]


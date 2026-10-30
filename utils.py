"""Helper functions and global settings"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import pandas as pd

from Polytope import Polytope


############################
##    GLOBAL VARIABLES    ##
############################

ANGLES = np.arange(0,360) * np.pi/180                               # 360 equally spaced angles (in radians)
UNIT_VECTORS = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])    # 2D unit vectors, shape (360, 2)
ALPHAS = np.linspace(0,1,41)                        # quantile levels, 0 to 1 in steps of 0.025

AIRPORTS_CSV = "data/airports.csv"                                  # raw: airport list with coordinates
FLIGHT_LIST_CSV = "data/flight_list.csv"                            # raw: flights between airports
CONNECTIONS_CSV = "data/flight_connections.csv"                     # output of `process_flights`
SCANDINAVIAN_AIRLINE = "8be5c854fd664bcb97fb543339f74770"           # airline ID used for the Scandinavian example
ATHENS_CSV = "https://s.ics.upjs.sk/~matej_uhrin/files/athens_data.csv"    # raw: air quality in Athens data


###########################
##    PLOTTING HELPERS   ##
###########################

def set_plotting_style():
    """Apply the matplotlib style used for all figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })


def save_figure(figure, path, fig_size=None, format="png", dpi=600, pad_inches=0.1):
    """Save a `figure` as in PNG or/and TIFF.

    `path` is given without extension; If `fig_size` is given, the figure is resized first.
    """
    if fig_size is not None:
        figure.set_size_inches(fig_size)

    if format == "png" or format == "both":
        figure.savefig(fname=f"{path}.png", dpi=dpi, pad_inches=pad_inches, bbox_inches="tight",
                   format="png")
    if format == "tiff" or format == "both":
        figure.savefig(fname=f"{path}.tiff", dpi=dpi, pad_inches=pad_inches, bbox_inches="tight",
                   format="tiff", pil_kwargs={"compression": "tiff_lzw"})

def make_pairplot_axes():
    """Create the empty 2x2 layout used by `PolytopeRandomSet.pairplot`."""
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    for a in ax.flat: a.tick_params(labelsize=9)
    ax[1,0].sharex(ax[0,0])
    ax[1,0].sharey(ax[1,1])
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    ax[0,1].axis("off")
    ax[0,0].set_ylim(-0.05, 1.05)
    ax[0,0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0,1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0,1].tick_params(axis='y', left=False, labelleft=False)
    ax[1,1].tick_params(axis='y', left=False, labelleft=False)

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="gist_rainbow_r"), ax=ax[0,1], location="bottom",
                        fraction=0.15, shrink=0.9, anchor=(0.5,0.5))
    cbar.ax.xaxis.tick_top()
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_tick_params(labelsize=9)
    cbar.ax.set_xlabel("$\\boldsymbol{\\alpha}$", fontsize=12, usetex=True)

    ax[0][0].set_ylabel("$\\mathsf P\\left(X\\subseteq K_\\alpha\\right)$", fontsize=13, usetex=True)
    ax[1][0].set_xlabel("$\\alpha$", fontsize=12, usetex=True)
    ax[1][0].set_ylabel("Volume of $K_\\alpha$", fontsize=13, usetex=True)
    ax[1][1].set_xlabel("$\\mathsf P\\left(X\\subseteq K_\\alpha\\right)$", fontsize=13, usetex=True)

    return fig, ax


###########################
##    CAPACITY HELPERS   ##
###########################

def capacity_point_levels(R, B):
    """For each row R[i] (length 360), return the largest k such that
    R[i] <= B[k-1].

    B (m x 360) is componentwise non-increasing in k, so the condition is
    monotone in k and the largest k is found by binary search.
    """
    N, m = R.shape[0], B.shape[0]
    lo = np.zeros(N, dtype=int)
    hi = np.full(N, m, dtype=int)
    while True:
        active = lo < hi
        if not active.any():
            return lo
        mid = (lo + hi + 1) // 2
        ok = (R - B[mid - 1]).max(axis=1) <= 1e-9
        lo = np.where(active & ok, mid, lo)
        hi = np.where(active & ~ok, mid - 1, hi)

def capacity_level_bounds(pts, starts, A, B, chunk=20000):
    """Cheap lower and upper bounds on the capacity level of each sample set.

    Parameters
    ----------
    pts : (total_vertices, d) array
        Vertices of all sample sets stacked on top of each other.
    starts : (n + 1,) array
        Start index of each set in `pts` (last entry = total_vertices).
    A : (360, d) array
        Common directions of the quantiles.
    B : (m_eff, 360) array
        Support values of the non-empty quantiles, non-increasing in the level.
    chunk : int
        Approximate number of vertices processed at once (memory control).

    Returns
    -------
    lower, upper : (n,) int arrays
        lower[j]: level reached by the best single vertex of set j;
        upper[j]: level reached by the componentwise minimum over its vertices.
        The true level lies in [lower[j], upper[j]].
    """
    n = len(starts) - 1
    lower, upper = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    blk = max(1, int(chunk // (len(pts) / n)))
    for s in range(0, n, blk):
        e = min(n, s + blk)
        M = pts[starts[s]:starts[e]] @ A.T            # (vertices of block, 360)
        seg = starts[s:e] - starts[s]
        lower[s:e] = np.maximum.reduceat(capacity_point_levels(M, B), seg)
        upper[s:e] = capacity_point_levels(np.minimum.reduceat(M, seg, axis=0), B)
    return lower, upper

def refine_level_by_lp(V, A, B, lo, hi):
    """Exact capacity level of one set with vertices V, searched in [lo, hi].

    Bisection on the level; each step is an LP feasibility test of whether
    some convex combination of the vertices of V lies in the quantile
    {x : A x <= B[level - 1]}.
    """
    G = A @ V.T                                       # (360, r), once per set
    r = G.shape[1]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        res = linprog(np.zeros(r), A_ub=G, b_ub=B[mid - 1] + 1e-9,
                      A_eq=np.ones((1, r)), b_eq=[1.0],
                      bounds=(0, None), method="highs")
        if res.status == 0:        # feasible
            lo = mid
        elif res.status == 2:      # infeasible
            hi = mid - 1
        else:
            raise RuntimeError(f"Linprog failed: {res.status} - {res.message}")
    return lo


###########################
##       MAP PLOTS       ##
###########################

def compute_map_extent(RS):
    """Return the map extent [lon_min, lon_max, lat_min, lat_max] covering all sample sets
    in the random set `RS`, with a 10% margin and a fixed height/width ratio of 0.75."""
    lons = [r.V.T[0] for r in RS.sample]
    lats = [r.V.T[1] for r in RS.sample]
    x0, x1, y0, y1 = np.min(lons), np.max(lons), np.min(lats), np.max(lats)
    w, h = (x1-x0) * 1.1, (y1-y0) * 1.1
    if h / w < 0.75: h = w * 0.75
    else: w = h / 0.75
    cx, cy = (x0+x1) / 2, (y0+y1) / 2
    return [cx - w/2, cx + w/2, cy - h/2, cy + h/2]

def draw_map(ax):
    """Draw the base map (ocean, land, coastlines, borders) on a cartopy axis."""
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=cfeature.COLORS["land"], zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.25, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), lw=0.3, linestyle=":", zorder=2)

def plot_flights_on_map(RS):
    """Plot all flights of the (flight-based) random set `RS` on a map."""
    fig, ax = plt.subplots(1,1, figsize=(5,5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    draw_map(ax)

    for flight in RS.sample:
        ax.plot(flight.V.T[0], flight.V.T[1], color="steelblue", alpha=0.3, linewidth=0.8, zorder=3)

    lons = [r.V.T[0] for r in RS.sample]
    lats = [r.V.T[1] for r in RS.sample]
    ax.scatter(lons, lats, s=1, color="navy", zorder=3)
    ax.set_extent(compute_map_extent(RS), crs=ccrs.PlateCarree())
    return ax

def plot_quantiles_on_map(RS, alphas=ALPHAS):
    """Plot the quantile sets of random set `RS` for the levels in `alphas` array."""
    fig, ax = plt.subplots(1,1, figsize=(5,5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    draw_map(ax)

    quantiles = RS.quantiles(alphas=alphas)
    for poly in quantiles.values():
        if poly.is_empty:
            continue
        poly.plot(ax=ax, patch_args={"fc":"seagreen", "alpha":0.5})

    ax.set_extent(compute_map_extent(RS), crs=ccrs.PlateCarree())
    return ax

def plot_flights_bagplot(RS):
    """Plot the bagplot of random set `RS` on a map."""
    fig, ax = plt.subplots(1,1, figsize=(5,5), subplot_kw=dict(projection=ccrs.PlateCarree()))
    draw_map(ax)
    RS.bagplot(ax=ax)
    ax.set_extent(compute_map_extent(RS), crs=ccrs.PlateCarree())
    return ax


###########################
##       MAP PLOTS       ##
###########################


######################################
##  DATA LOADING AND PREPROCESSING  ##
######################################

def load_airports():
    """Read the European airports with known coordinates."""
    cols = ["ident", "continent", "latitude_deg", "longitude_deg"]
    df = pd.read_csv(AIRPORTS_CSV, usecols=cols, dtype=str, keep_default_na=False)
    df = df.apply(lambda c: c.str.strip())
    df = df[(df["ident"] != "") & (df["continent"] == "EU") & (df["latitude_deg"] != "") & (df["longitude_deg"] != "")]
    return df.drop_duplicates("ident", keep="last").set_index("ident")[["latitude_deg", "longitude_deg"]]


def process_flights():
    """Attach airport coordinates to the raw flight list and save the result."""
    airports = load_airports()
    cols = ["date", "name_adep", "name_ades", "adep", "ades", "airline"]
    flights = pd.read_csv(FLIGHT_LIST_CSV, usecols=cols, dtype=str, keep_default_na=False)
    total = len(flights)
    flights["adep"] = flights["adep"].str.strip()
    flights["ades"] = flights["ades"].str.strip()

    dep = airports.rename(columns={"latitude_deg": "adep_latitude_deg", "longitude_deg": "adep_longitude_deg"})
    arr = airports.rename(columns={"latitude_deg": "ades_latitude_deg", "longitude_deg": "ades_longitude_deg"})
    flights = (flights.merge(dep, left_on="adep", right_index=True).merge(arr, left_on="ades", right_index=True)
               .sort_index())

    out_cols = ["date", "name_adep", "name_ades", "adep_latitude_deg", "adep_longitude_deg",
                "ades_latitude_deg", "ades_longitude_deg", "airline"]
    flights[out_cols].to_csv(CONNECTIONS_CSV, index=False, encoding="utf-8")
    print(f"Processed {len(flights)} out of {total} flights -> {CONNECTIONS_CSV}")


def load_connections():
    """Read `CONNECTIONS_CSV`.

    Returns
    -------
    DataFrame
        Columns date, name_adep, name_ades, adep_lat, adep_lon, ades_lat,
        ades_lon (floats) and airline.
    """
    df = pd.read_csv(CONNECTIONS_CSV, dtype=str, keep_default_na=False)
    df = df.rename(columns={"adep_latitude_deg": "adep_lat", "adep_longitude_deg": "adep_lon",
                            "ades_latitude_deg": "ades_lat", "ades_longitude_deg": "ades_lon"})
    coords = ["adep_lat", "adep_lon", "ades_lat", "ades_lon"]
    df[coords] = df[coords].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=coords)


def get_flight_data(airline=None, adep=None, ades=None, date1=None, date2=None):
    """Load flights as segments (longitude, latitude) between the two airports.

    Parameters
    ----------
    airline : str, optional
        Airline identifier; all airlines if None.
    adep, ades : str, optional
        Name of the departure / arrival airport; all airports if None.
    date1, date2 : str, optional
        First and last date (inclusive, format "YYYY-MM-DD"); None means no limit.

    Returns
    -------
    list of Polytope
        One segment (a polytope with two vertices) per flight.
    """
    df = load_connections()

    if airline is not None:
        df = df[df["airline"] == airline]
    if adep is not None:
        df = df[df["name_adep"] == adep]
    if ades is not None:
        df = df[df["name_ades"] == ades]
    if date1 is not None:
        df = df[df["date"] >= date1]
    if date2 is not None:
        df = df[df["date"] <= date2]
    df = df[["adep_lat", "adep_lon", "ades_lat", "ades_lon"]]

    return [Polytope(V=[[lon1, lat1], [lon2, lat2]]) for lat1, lon1, lat2, lon2 in df.values]


def hull_vertices(points):
    """Vertices of the convex hull of a set of 2D points."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3: return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:  # QhullError for collinear points
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        return pts[[order[0], order[-1]]]


def load_athens_hulls(x, y, date_from, date_to):
    """Daily convex hulls of the (x, y) measurements, one per station and day.

    Parameters
    ----------
    x, y : str
        Names of the two measured variables (temperature and ozone by default).
    date_from, date_to : str
        Exclusive limits of the date range, format "YYYY-MM-DD".

    Returns
    -------
    list of dict
        One dict per station-day with keys "station", "date",
        "all_data_points" ((k, 2) array) and "hull_polytope" (Polytope).
    """
    df = pd.read_csv(ATHENS_CSV).dropna()
    df = df[(df["Date"] > date_from) & (df["Date"] < date_to)]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    hulls = []
    for station, g in df.groupby("station_name"):
        for date, h in g.groupby("Date"):
            pts = h[[x, y]].to_numpy()
            if len(pts) < 3:
                continue
            hulls.append({"station": station,
                          "date": date,
                          "all_data_points": pts,
                          "hull_polytope": Polytope(V=hull_vertices(pts))})
    return hulls


if __name__ == "__main__":
    process_flights()
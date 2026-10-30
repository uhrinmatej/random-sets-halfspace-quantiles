## random-sets-halfspace-quantiles
 
Supplementary code for the paper **"Halfspace-based Quantiles of Random Convex Sets"**.
 
The repository computes halfspace-based quantiles of random convex sets (random polytopes and random intervals)
from a finite sample, together with the related functionals and diagnostic plots, and reproduces the examples
and figures of the paper.

## Repository structure
 
| File | Content                                                             |
|---|---------------------------------------------------------------------|
| `Polytope.py` | extension of `pycvxset` Polytope class                              |
| `PolytopeRandomSet.py` | Python class representing random polytope given by a sample         |
| `IntervalRandomSet.py` | Python class representing random 1D interval given by a sample      |
| `utils.py` | Helper functions and global settings                                |
| `example_ball.ipynb` | Ball of fixed radius with a random center                           |
| `example_uniqueness.ipynb` | Random ball and random line with the same quantile system           |
| `example_rectangle.ipynb` | Random rectangle, comparison with coverage-based quantiles          |
| `rand_interval.ipynb` | Random intervals, comparison with coverage-based quantiles          |
| `data_athens.ipynb` | Real data: daily convex hulls of air quality measurements in Athens |
| `data_flights.ipynb` | Real data: flight trajectories                                      |
| `data/` | Data files (see below)                                              |
| `images/` | Figures produced by the notebooks                                   |
 
## Installation
 
Python 3.10 or newer is required.
 
```bash
git clone https://github.com/uhrinmatej/random-sets-halfspace-quantiles.git
cd random-sets-halfspace-quantiles
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
Notes:
 
- Figure labels are typeset with LaTeX (`usetex=True`), so a LaTeX installation with the `amsmath` package must be
available (for example TeX Live or MiKTeX).
- `utils.py` imports `cartopy`. Cartopy downloads the Natural Earth map data on first use, so the first run of 
`data_flights.ipynb` needs an internet connection.

## Running the notebooks
 
Start JupyterLab **from the repository root** (the notebooks use the relative paths `data/` and `images/`):
 
```bash
jupyter lab
```
 
- All random draws use fixed seeds (2026), so rerunning the cells in order reproduces the figures. The notebooks share
one random generator, so run their cells from top to bottom.
- To export the figures to `images/`, set `SAVE_FIGURES = True` in the first code cell of a notebook.
- The notebooks are stored with their outputs, so the figures can be viewed without running anything.


## Data

### Flights

Source: [1], licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| File | Content                                                                                                                                                |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data/flight_list.csv` | Copy of `flight_list.csv` from the dataset above                                                                                                       |
| `data/airports.csv` | Airport list with coordinates Airport list with coordinates (downloaded from [OurAirports](https://ourairports.com/data/)) | |
| `data/flight_connections.csv` | Derived: airport coordinates attached to each flight. Regenerate with `python utils.py`                                                                |

### Athens air quality

Source: [2], licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Data set is read directly from the URL stored in `ATHENS_CSV` in `utils.py`, so
`data_athens.ipynb` needs an internet connection. To work offline, download the file to `data/` and set
`ATHENS_CSV = "data/athens_data.csv"`.



## Acknowledgements

The code uses the package [pycvxset](https://github.com/merlresearch/pycvxset) [3] for polytope operations
 and also package [data-depth](https://data-depth.github.io) [4] for halfspace depth computation. Other used packages are NumPy, SciPy, pandas,
Matplotlib and Cartopy.

[1] Spinielli, E., Sun, J., Olive, X., Strohmeier, M. et al. (2025).
*Dataset for EUROCONTROL PRC 2024 Data Challenge on Aircraft Take-off Weight Estimation*.
4TU.ResearchData. [doi:10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4](https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4)
<br>[2] Angelis, G.-F. (2024). *Regional Datasets for Air Quality Monitoring in European Cities*.
Zenodo. [doi:10.5281/zenodo.11220965](https://doi.org/10.5281/zenodo.11220965)
<br>[3] A. P. Vinod, *pycvxset: A Python package for convex set manipulation*, ACC 2025.
<br>[4] P. Mozharovskyi et al., *data-depth: Python library for data depth*, https://data-depth.github.io.



<!--## Citation
 
If you use this code, please cite the paper:
 
> *Halfspace-based Quantiles of Random Convex Sets*.
 
TODO: add authors, journal / arXiv reference and BibTeX once available. -->
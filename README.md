### Embryo development simulation with subcellular element method and optimal transport
A study of early mammalian embryo development using a data-driven model based on subcellular element method and optimal transport.

The main Python script for simulation has the following dependencies: ``torch`` (1.3.1); ``numba`` (0.48.0 with cuda); ``scipy``; ``numpy``; ``pandas``; the package in the folder ``dot``. The plotting utilities for the embryo depends on ``mayavi`` and ``matplotlib``.

To reproduce the results:
  1. Preprocess data in ``data_preprocessing``.
  2. Run the script ``semot_simulation.py``.
  3. Analyze results in ``result_analysis``.

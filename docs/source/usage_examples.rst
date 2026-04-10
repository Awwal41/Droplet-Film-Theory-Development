Usage Examples and Tutorials
==============================

Code for training and evaluation lives under ``train_eval/``. Add ``train_eval`` to your Python path
(or run notebooks/scripts with ``train_eval`` as the working directory) so imports like
``from src.dfm_src import DFT`` resolve.

.. code-block:: bash

   cd train_eval
   python -c "from src.dfm_src import DFT; from src.utils import Helm; print('OK')"


DFM / DFT — ordinal droplet-film model
--------------------------------------

``DFT`` predicts **loading class** (0 = Unloaded, 1 = Near L.U., 2 = Loaded) from
well features and measured gas rate using a physics-based :math:`Q_{\mathrm{cr}}` and ordinal
thresholds — not a direct regression on a ``Qcr`` column.

**Data:** CSV with the usual feature columns, **Gasflowrate**, and **Test status** (Unloaded /
Near L.U / Loaded). See ``datasets/processed_well_data.csv`` for a reference layout.

.. code-block:: python

   import numpy as np
   from src.dfm_src import DFT
   from src.utils import Helm

   data_path = "datasets/processed_well_data.csv"  # or pathlib.Path
   helm = Helm(
       path=data_path,
       includ_cols=[
           "Dia", "Dev(deg)", "Area (m2)", "z", "GasDens", "LiquidDens",
           "g (m/s2)", "P/T", "friction_factor", "critical_film_thickness",
       ],
       test_size=0.2,
       scale=False,
   )

   dft = DFT(seed=42, feature_tol=1.0, dev_tol=1e-3, multiple_dev_policy="max")
   dft.fit(helm.X_train, helm.gsflow_train, helm.loading_train)

   pred = dft.predict(helm.X_test, helm.gsflow_test)
   acc = float(np.mean(pred == helm.loading_test))
   print("Test accuracy:", acc)

For the **same CSV split** as QLattice / TabPFN (dynamic operational margins), use
``Helm.from_dynamic_bounds_csv`` and map labels with ``loading_dynamic_bounds_to_ordinal`` before
``DFT.fit`` — see ``train_eval/DFM.ipynb``.


QLattice (Feyn) and TabPFN pipelines
--------------------------------------

Symbolic regression and TabPFN share filtering, stratified split, and **dynamic-bounds** metrics in
``train_eval/src/qlattice_pipeline.py`` and ``train_eval/src/tabpfn_pipeline.py``.

- **Recommended mode for a self-contained run:** ``qcr_labels`` (label-proxy targets built inside the
  pipeline; no external ``.npz``).
- **CLI** (from repository root or ``train_eval``):

.. code-block:: text

   python train_eval/qlattice_standalone.py --mode qcr_labels --plot-dir ./plots_out
   python train_eval/tabpfn_standalone.py --mode qcr_labels --qcr-proxy-margin-scale 2.5 --no-plots

- **Programmatic** (same as the standalone scripts):

.. code-block:: python

   from src.utils import run_qlattice_from_argv, run_tabpfn_from_argv, default_qlattice_csv_path

   csv_path = default_qlattice_csv_path()
   run_tabpfn_from_argv(["--csv", str(csv_path), "--mode", "qcr_labels", "--no-plots"])

Notebooks: ``train_eval/QLattice.ipynb``, ``train_eval/TabPFN.ipynb``.


XGBoost (example)
-----------------

Use scaled features from ``Helm`` when ``scale=True``:

.. code-block:: python

   import xgboost as xgb
   from sklearn.metrics import accuracy_score
   from src.utils import Helm

   includ_cols = [
       "Dia", "Dev(deg)", "Area (m2)", "z", "GasDens", "LiquidDens",
       "g (m/s2)", "P/T", "friction_factor", "critical_film_thickness",
   ]
   helm = Helm(
       path="datasets/processed_well_data.csv",
       includ_cols=includ_cols,
       test_size=0.2,
       scale=True,
   )
   clf = xgb.XGBClassifier(random_state=42)
   clf.fit(helm.X_train_rdy, helm.loading_train)
   pred = clf.predict(helm.X_test_rdy)
   print("Accuracy:", accuracy_score(helm.loading_test, pred))


PySINDy and PySR (illustrative)
-------------------------------

The sections below are **illustrative** templates. They are not pinned to the current ``Helm`` / DFM
API; adapt shapes and targets to your experiment.

.. code-block:: python

   # pip install pysindy pysr
   # Use scaled arrays from Helm; ensure targets match your task (regression vs classification).


Plots from pipelines
--------------------

QLattice / TabPFN can write figures when you pass ``--plot-dir`` (and omit ``--no-plots``). Figures
use :func:`train_eval.src.qlattice_pipeline.plot_turner_qmeas_vs_qrate` and
``plot_residual_qminus_qmeas`` (dynamic δ bands). For headless servers set ``MPLBACKEND=Agg``.


Model comparison (sklearn-style)
--------------------------------

Compare predictions on the **same** ``helm.loading_test`` when models output class labels compatible
with Helm’s encoding (0/1/2).


Further reading
---------------

* :doc:`api_reference` — classes and parameters
* :doc:`installation_guide` — environment setup
* :doc:`troubleshooting` — common issues

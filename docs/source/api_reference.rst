API Reference
=============

Technical overview of the Python package under ``train_eval/src/``. Import paths assume the
``train_eval`` directory is on ``sys.path`` (e.g. run code from ``train_eval`` or install the
project in editable mode with that layout).

Module layout
-------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - ``src/dfm_src.py``
     - ``DFT`` — droplet-film ordinal model (``fit``, ``predict``, ``predict_proba``).
   * - ``src/utils.py``
     - ``Helm`` — load CSV, stratified split, scaling, CV, and helpers; ``QLatticeWrapper`` — Feyn QLattice for gas / Q_cr modes with dynamic-bounds evaluation; CLI helpers ``run_qlattice_from_argv``, ``run_tabpfn_from_argv``, ``default_qlattice_csv_path``.
   * - ``src/qlattice_pipeline.py``
     - Standalone QLattice script: data load/split, Turner-style metrics, plotting helpers ``plot_turner_qmeas_vs_qrate``, ``plot_residual_qminus_qmeas``.
   * - ``src/tabpfn_pipeline.py``
     - TabPFN regression with the same split and metrics as ``qlattice_pipeline``.

``DFT`` (``src.dfm_src``)
-------------------------

Ordinal classifier: fits physics parameters :math:`p_1,\ldots,p_5`, thresholds :math:`\tau_1,\tau_2`,
and per-sample :math:`\alpha` by minimizing ordinal cross-entropy on log-ratio
:math:`\log(Q_{\mathrm{cr,pred}}/Q_{\mathrm{meas}})`.

**Constructor**

.. code-block:: text

   DFT(seed=42, feature_tol=1.0, dev_tol=1e-3, multiple_dev_policy="max")

**Main methods**

- ``fit(X, gsflow, loading)`` — ``X`` shape ``(n, 10)``, ``loading`` integers ``{0,1,2}``.
- ``predict(X, gsflow)``, ``predict_proba(X, gsflow)`` — gas flow required for the ratio.

Feature column order matches ``DFT._eq`` (Dia, Dev(deg), Area (m2), z, GasDens, LiquidDens,
g (m/s2), P/T, friction_factor, critical_film_thickness).

``Helm`` (``src.utils``)
------------------------

Loads a CSV, stratifies on loading labels derived from **Test status**, and exposes
``X_train``, ``X_test``, ``gsflow_train``, ``gsflow_test``, ``loading_train``, ``loading_test``,
and scaled copies ``X_train_rdy``, ``X_test_rdy`` when ``scale=True``.

**Factory:** ``Helm.from_dynamic_bounds_csv(path, seed=42, test_size=0.2)`` returns an instance
aligned with ``qlattice_pipeline.load_and_split`` (labels ``-1/0/1`` for Unloaded / Near / Loaded).

``QLatticeWrapper`` (``src.utils``)
------------------------------------

Configured for ``mode`` in ``gas`` | ``qcr_labels`` | ``qcr_custom``. For operational loading
metrics, use ``QLatticeWrapper.fit_dynamic_bounds`` (not sklearn ``fit(X, y)`` on raw classes).

Plotting (``src.qlattice_pipeline``)
--------------------------------------

- ``plot_turner_qmeas_vs_qrate(...)`` — cross-plot of measured gas rate vs predicted rate with
  dynamic margin bands; skips cleanly if arrays are empty or all non-finite.
- ``plot_residual_qminus_qmeas(...)`` — residual vs :math:`Q_{\mathrm{meas}}` with the same bands.
- ``plot_qcr_x_qmeas_y_multimethod(...)`` — **x** = predicted :math:`Q_{\mathrm{cr}}` for each of
  several methods, **y** = :math:`Q_{\mathrm{meas}}`; optional status shading, 1:1 line, and
  near-diagonal highlights.

Data columns
------------

Processed well CSVs should include the ten physics features above, **Gasflowrate**, and **Test
status** (or the filtered columns expected by ``qlattice_pipeline`` — see ``TURNER_FEATURE_COLS``).

Support
-------

For runnable examples see :doc:`usage_examples` and notebooks under ``train_eval/``.

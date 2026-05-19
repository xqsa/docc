# Verification

This workspace has been rolled back to the original `hcc_es_original` implementation.

Current verification status:

- `python -m py_compile HCC_SRC/HCC-ES.py HCC_SRC/run_pypop7_baselines.py HCC_SRC/experiment_protocols.py` passes.
- `python -m unittest discover -s tests -p 'test_*.py' -v` passes with `26` tests.
- `python HCC_SRC/HCC-ES.py --problems E4 --seeds 1 --tfes 20 --method hcc_es_original --output-dir tmp/hcc-rollback-smoke` completes successfully and writes `summary.csv`, `run_details.csv`, and `diagnostics.csv`.

Removed from the runnable codebase:

- `CHCFR`
- `OHR`
- `OHBCD`
- `U-OHBCD`
- risk-gated ownership variants
- graph-risk-gated variants

Historical experiment outputs may still exist under `HCC_SRC/result/` as archived artifacts, but they are no longer part of the active implementation surface.

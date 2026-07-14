# SIESTA output regression fixtures

These are deliberately minimized output excerpts: they retain the table
headings, column order, repeated-result behavior, and convergence markers that
the parser consumes, without checking a large binary-specific SIESTA log into
the test suite.

- `siesta_4_1_mulliken.out` covers the legacy spin-resolved `Qatom` layout
  documented in the SIESTA MnO DFT+U tutorial:
  <https://docs.siesta-project.org/projects/siesta/en/latest/tutorials/advanced/dft%2Bu/index.html>
- `siesta_4_1_hirshfeld_net.out` covers the `Hirshfeld Net Atomic
  Populations:` heading used by SIESTA 4.x population output.
- `siesta_5_atomic_populations.out` covers the common signed `Sz` atomic
  population layout documented in the SIESTA 5.x reference manual:
  <https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html#charge-populations>

The fixtures include more than one energy/spin/table occurrence where useful,
so tests verify that the final reported state is selected.

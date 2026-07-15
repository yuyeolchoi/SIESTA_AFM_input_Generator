# siesta-afm

English · **[한국어](README.ko.md)**

`siesta-afm` is a Python CLI that selects magnetic atoms from CIF, XYZ, POSCAR/CONTCAR, SIESTA XV, and FDF structures and builds a `%block DM.InitSpin` initial-spin arrangement for SIESTA. It never reorders the input atoms, and the SIESTA output indices are always one-based.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
```

Oxidation-state guessing, visualization, and the GUI are separate optional dependencies.

```bash
python -m pip install -e ".[oxidation,plot,gui,yaml]"
```

## Quick start

```bash
siesta-afm analyze examples/CuO_111_slab.cif \
  --magnetic-species Cu --slab

siesta-afm generate examples/CuO_111_slab.cif \
  --magnetic-species Cu \
  --method layer \
  --axis z \
  --moment 0.5 \
  --slab \
  --output afm_spin.fdf

siesta-afm validate afm_spin.fdf \
  --structure examples/CuO_111_slab.cif \
  --magnetic-species Cu --slab

siesta-afm plot examples/CuO_111_slab.cif \
  --spin-file afm_spin.fdf \
  --output afm_pattern.png --slab
```

By default `plot` shows the spin sign in red/blue. To compare different site moments on a continuous color scale, use `--color-mode value`. That mode adds a zero-centered symmetric color range and an `initial spin (μB)` colorbar, and `--up-color` / `--down-color` are ignored.

To insert the result into an existing SIESTA input:

```bash
siesta-afm patch examples/input.fdf \
  --spin-file examples/afm_spin.fdf \
  --output input_afm.fdf
```

Without `--in-place`, the original FDF is never overwritten. Adding `--backup` writes a `.bak` copy next to the original.

## Generation methods

- `alternating-index`: assigns `+ - + -` within the selected magnetic-atom list only.
- `layer`: groups coordinates along `--axis` using `--layer-tolerance` and alternates the sign per layer.
- `checkerboard`: two-colors the in-plane nearest-neighbor graph of the `--plane xy|xz|yz` plane.
- `neighbor-bipartite`: builds the PBC minimum-image nearest-neighbor graph and colors the two sublattices.
- `graph-coloring`: uses DSATUR proper coloring to build up to k sublattice candidates and maps a collinear spin to each color.
- `propagation-vector`: sets the sign from `sign(cos(2π q·r + phase))`.
- `manual-groups`: uses `--up-atoms`, `--down-atoms`, or a YAML `--group-file`.
- `by-species`: splits distinct-element sublattices with `--up-species` and `--down-species`.
- `by-coordination`: splits Td/Oh sublattices by each magnetic atom's first anion-shell coordination number.
- `random`: produces reproducible random initial signs with `--seed`. This is not a physical magnetic-ordering model.

Examples:

```bash
siesta-afm generate structure.cif \
  --magnetic-species Ni Co \
  --method neighbor-bipartite \
  --moment Ni=1.0 Co=1.5 \
  --cutoff auto \
  --output afm_spin.fdf

siesta-afm generate structure.cif \
  --magnetic-species Cu \
  --method propagation-vector \
  --q-vector 0.5 0.5 0.5 \
  --moment 0.5

siesta-afm generate structure.cif \
  --magnetic-species Cu \
  --method manual-groups \
  --up-atoms 2,5,8,11 \
  --down-atoms 3,6,9,12 \
  --moment 0.5

siesta-afm generate spinel.cif \
  --magnetic-species Fe \
  --method by-coordination \
  --anion-species O \
  --up-coordination 6 --down-coordination 4 \
  --moment Fe@6=4.0 Fe@4=3.0

siesta-afm generate examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co \
  --method by-coordination \
  --moment Co@4=3.0 Co@6=0.5

siesta-afm generate inverse_spinel.cif \
  --magnetic-species Ni Co \
  --method by-coordination \
  --anion-species O \
  --moment Ni@6=2.0 Co@4=2.0 Co@6=0.0
```

`--moment 0.5` applies the same magnitude to every selected element, while `--moment Cu=0.5 Ni=1.0` sets per-element values. `Element@CN=value` distinguishes different coordination environments of the same element. The resolution priority is site CSV > `Element@CN` > `Element` > global value. The CSV given to `--site-moment-file moments.csv` requires at least `atom_index,moment` columns and may optionally include `element,oxidation_state` columns.

For `by-species`, the union of up/down must exactly match `--magnetic-species`. That method suits distinct-element sublattices such as Ni/Co, but it cannot separate an inverse spinel in which one element occupies both Td and Oh sites — use `by-coordination` there. `by-coordination` auto-detects the anion when exactly one of O, S, Se, Te, N, F, Cl is present, and requires `--anion-species` when several candidates exist. It counts distinct periodic images of the same basis anion as separate neighbors, and the default classification is up CN=6, down CN=4. Use `--anion-cutoff` and `--coordination-tolerance` to tune the decision.

In the inverse-spinel command above, the default coordination sublattices produce
Ni(Oh)=+2 μB, Co(Td)=−2 μB, and low-spin Co(Oh)=0. Using one `Co=value`
for both CN=4 and CN=6 sites is allowed but now emits a warning because it cannot
represent those two Co sublattices independently.

The Co₃O₄ example uses the public-domain structure data from [COD 1538531](https://www.crystallography.net/cod/1538531.html) (Roth, 1964). Its Co coordination distribution is 8 atoms with Td CN=4 and 16 atoms with Oh CN=6.

The propagation-vector `q` is in fractional coordinates of the input cell. In a supercell, `q` must be scaled down to represent the same physical periodicity. The A/C/G presets are selected with `--afm-type A`, `C`, `G` respectively and cannot be combined with a user `--q-vector`. Layers not parallel to an axis, such as NiO's (111) AFM-II, are generated with `--method layer --layer-direction 1 1 1`.

A YAML configuration is read with `--moment-config moments.yaml`.

```yaml
moments:
  Cu: 0.5
  Ni: 1.0
  Co: 1.5
```

Oxidation-state guessing never runs by default. It runs only when the user passes `--guess-oxidation-states` and the optional `pymatgen` dependency is installed, and it warns that the result is an estimate.

## Periodic boundaries, slabs, and adsorbates

You can use the input file's PBC or override it explicitly.

```bash
--slab                 # xy periodic, z nonperiodic
--periodic-axes xy     # the same setting, made explicit
--periodic-axes xyz    # fully 3D periodic structure
```

Because magnetic elements must be selected with `--magnetic-species`, adsorbates such as C, H, O, Cs are automatically excluded from the magnetic graph. Additional exclusion ranges are specified with `--exclude-atoms 217-228` or `--adsorbate-indices 217,218,219`.

## Frustrated / non-bipartite graphs

If the `neighbor-bipartite` graph is not bipartite, the program does not fabricate an arbitrary two-sublattice result — it exits with an error. In that case, consider layer, propagation vector, manual groups, or a different cutoff.

`--allow-frustrated` explicitly permits an iterative Max-Cut heuristic that increases the number of opposite-sign edges. Such a result carries the following scientific warning:

> The generated spin assignment is a heuristic initial state for a frustrated magnetic network. It is not guaranteed to represent the experimental magnetic ground state.

When the graph splits into two or more connected components, only the alternating sign inside each component is determined by the graph. The relative sign between components is a deterministic convention based on the lowest atom index and has no physical meaning. In that case the program warns with the component count and sizes and suggests tuning `--neighbor-cutoff` to include interlayer superexchange or considering the `layer` / `propagation-vector` methods. A layer slab with an odd number of magnetic layers along a nonperiodic direction is not an error but an informational warning that it is an uncompensated AFM slab.

`graph-coloring` is a multi-sublattice initial-candidate generator that applies DSATUR to a non-bipartite graph. `--max-colors` defaults to 4, and you can assign per-color signs with `--color-spins "+1,-1,0"` or use `--balance-colors` to pick the color-sign permutation that minimizes the absolute sum of the actual initial moments resolved from `--moment`, per-element moments, or a site-moment file. A proper coloring only avoids equal colors on adjacent atoms; it does not minimize energy. If a collinear energy candidate for a frustrated lattice is the goal, `--allow-frustrated` max-cut is more appropriate. In `enumerate`, the color-spin permutation is varied by the attempt seed to diversify candidates.

## Analysis and validation

`analyze` reports magnetic-atom distance shells, the automatic cutoff, graph size / connectivity / bipartiteness, and the number of layers, and supports `--json analysis.json`.

`validate` checks for duplicate / out-of-range indices, nonzero spin on atoms that are not a selected magnetic element, up/down counts, and the net spin. When `--structure` is given, it also computes the nearest-neighbor antiparallel fraction (`AFM score`), connected components, and the per-layer distribution.

## Multiple candidates and SIESTA job arrays

```bash
siesta-afm enumerate structure.cif \
  --magnetic-species Cu \
  --moment 0.5 \
  --methods layer,checkerboard,frustrated \
  --n-configs 8 \
  --output-dir afm_configs

siesta-afm prepare-array examples/input.fdf \
  --configs afm_configs \
  --template input_setting.fdf \
  --output-dir siesta_afm_jobs

siesta-afm collect-results siesta_afm_jobs
```

`enumerate` removes identical patterns and, by default, treats a global sign inversion as the same candidate. Use `--keep-global-spin-inversion` to keep the two inversions separate. The `manifest.csv` records the method, up/down counts, net spin, and AFM score.

`collect-results` reads energy, final net spin, per-atom Mulliken/Hirshfeld spin, sign-retention fraction, collapse, and convergence markers from the common SIESTA `.out`/`.log` representations in each folder and writes `results.csv`. Different SIESTA versions may use different output wording, which can require extending the regular expressions.

## Running the GUI

First install the GUI optional dependencies.

```bash
python -m pip install -e ".[gui]"
```

On Windows, double-clicking `run_gui.bat` at the repository root opens a Tkinter desktop window rather than a browser. The launcher uses `.venv\Scripts\python.exe` first, and falls back to `python` on PATH. If Tkinter or matplotlib is unavailable, it keeps the window open while showing the install command. No browser-based server is required.

From a terminal, the following two entry points launch the same GUI.

```bash
python -m siesta_afm.gui
siesta-afm-gui
```

The GUI provides structure-file selection; magnetic element / method / moment / cutoff / layer settings; species and coordination sublattices; the A/C/G presets and an arbitrary layer direction; a rotatable and zoomable 3D preview; and graph analysis with a DM.InitSpin preview. Parameter changes are applied automatically with a 400 ms debounce, and `Live update` can be turned off. Existing spin files can also be opened on top of the current structure.

For `by-coordination`, `Suggest` detects the actual `Element@CN` combinations and
replaces an empty or single global moment with an editable site-specific template.
The detected combinations and atom counts remain visible below the field. `Site
moment file` accepts the same CSV as CLI `--site-moment-file` and keeps the priority
site CSV > `Element@CN` > `Element` > global.

After generation, the `Sites` tab lists every magnetic atom in input order with its
element, CN, sublattice, sign, and moment. Its footer shows `n_up`, `n_down`,
`n_zero`, and the net initial moment. The spin-file viewer populates the same table,
with CN and sublattice shown as `-` because those values are not stored in a spin
block.

The analysis corresponding to the CLI `analyze` is not a separate button — it runs automatically on generation and live updates and appears in the right-hand `Analysis` tab as distance shells, cutoff, connectivity, bipartiteness, and the number of layers.

Export can save the DM.InitSpin block, a patched SIESTA input that never overwrites the original, and an XYZ/CIF structure with the initial magmom included. The CLI is the reference scientific implementation, and the GUI controller uses the same core functions.

## Input and index preservation

ASE reads CIF, XYZ, and POSCAR/CONTCAR. XV has a dedicated parser used when ASE fails. The FDF parser handles the following blocks and recursive `%include`.

- `ChemicalSpeciesLabel`
- `AtomicCoordinatesAndAtomicSpecies`
- `LatticeVectors`

No input is ever sorted by element or by coordinate. The internal `ase_index` is the zero-based original order, and `siesta_index` is the one-based index in that same order.

## Important scientific caveats

1. `DM.InitSpin` values are only an initial guess, not the final local magnetic moments.
2. The spin arrangement can change after SCF.
3. A two-sublattice AFM is not appropriate for every oxide.
4. CuO(111), triangular surfaces, and spinel structures can be frustrated magnetic networks.
5. When an experimental or literature magnetic ordering is known, it should take precedence.
6. Multiple AFM/FM initial states should be computed and compared by final total energy.
7. The magnetic ground state can depend on the U value, basis, pseudopotential, and slab termination.

The small structures in `examples/` are for confirming CLI behavior and do not substitute for a converged surface-calculation model. In particular, `NiCo2O4_311_slab.cif` is a multi-element input/output demo, not a real spinel structure with constructed Td/Oh coordination, so it must not be used as a `by-coordination` validation model.

## Tests

```bash
python -m pytest
```

The tests cover FDF/include/patch, 1D / square / triangle / disconnected graphs, slab PBC, layer clustering, ordering, the one-based writer, and validation error detection.

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

Oxidation-state guessing, symmetry-aware candidate deduplication,
visualization, and the GUI are separate optional dependencies.

```bash
python -m pip install -e ".[oxidation,symmetry,plot,gui,yaml]"
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

Use `--filter-elements Ni Co` to show spin colors and arrows only for selected
elements; all other atoms remain visible as gray nonmagnetic markers. Add
`--show-bonds` to draw ASE covalent-radius bonds and adjust their detection with
`--bond-radius-scale` (default `1.0`). Bonds that cross a periodic cell boundary
are not displayed. Non-collinear arrows follow the `DM.InitSpin` theta/phi
directions, while up/down colors deliberately continue to use the sign of the
scalar moment (so a purely in-plane moment may still use the spin-up color).

To insert the result into an existing SIESTA input:

```bash
siesta-afm patch examples/input.fdf \
  --spin-file examples/afm_spin.fdf \
  --output input_afm.fdf
```

Without `--in-place`, the original FDF is never overwritten. Adding `--backup` writes a `.bak` copy next to the original.

## Complete SIESTA starting input

`make-input` combines the order-preserving structure, the selected AFM method,
`DM.InitSpin`, basis and pseudopotential requirements, PBE/SCF settings, a k-grid,
and an optional DFT+U block in one FDF file. For example, an inverse-spinel
starting point can be generated with:

```bash
siesta-afm make-input inverse_spinel.cif \
  --magnetic-species Ni Co \
  --method by-coordination \
  --anion-species O \
  --moment Ni@6=2.0 Co@4=2.0 Co@6=0.0 \
  --split-species-by-coordination \
  --hubbard-u Ni@6=6.0 Co@4=3.0 Co@6=5.0 \
  --output input.fdf
```

The coordinate rows and `DM.InitSpin` indices retain exactly the input atom
order. Without coordination splitting, existing FDF/XV species IDs are retained;
other formats receive species IDs in first-appearance order. The automatic
k-grid is
`ceil(30 Ang / |a_i|)` on each periodic axis and 1 on each nonperiodic axis.
Use `--kgrid N1 N2 N3` or `--kgrid-cutoff K` to change it, and
`--basis-size SZ|SZP|DZ|DZP|TZP` to select the starting basis.

By default, supported selected magnetic elements receive the Materials Project
oxide-calibrated effective U values in the SIESTA `LDAU.proj` syntax. Override
them with `--hubbard-u Element=value`, disable the block with `--no-lda-u`, or
select the current `DFTU.Proj` spelling with `--dftu-keyword dftu`. SIESTA uses
the Dudarev combination `U_eff = U - J` for this collinear setup, so the template
writes `U=U_eff` and `J=0`. DFT+U is applied per SIESTA species, not independently
per `(element, CN)` row; the command warns when one element occupies multiple
coordination sublattices.

For `--method by-coordination`, the opt-in
`--split-species-by-coordination` flag assigns separate, contiguous species IDs
to a magnetic element's distinct CN sites without changing coordinate-row order.
It reuses SIESTA labels such as `Co_2` and `Co_3`, enables CN-specific overrides
such as `--hubbard-u Co@4=3.0 Co@6=5.0`, and adds CN/geometry comments to
`ChemicalSpeciesLabel`. A plain `Co=3.0` remains valid and applies the same U to
all split Co species. Each generated label needs its own pseudopotential filename,
so copies such as `Co_2.psf` and `Co_3.psf` (or matching PSML files) must be placed
in the run directory. Without the flag, output remains unchanged and `@CN`
Hubbard-U overrides are rejected. The GUI exposes the same opt-in checkbox only
while `by-coordination` is selected.

The generated file is only a starting template. It is not publication-ready
until the pseudopotentials, basis, MeshCutoff, k-grid, SCF settings, Hubbard U,
and final magnetic state have been validated and converged. Current SIESTA
documentation recommends tested PSML data such as Pseudo-Dojo and documents
both `DFTU.Proj` and the `LDAU.Proj` alias. See the
[SIESTA pseudopotential guidance](https://siesta-project.org/siesta/Documentation/Pseudopotentials/),
[SIESTA DFT+U reference](https://docs.siesta-project.org/projects/siesta/en/stable/reference/siesta.html),
and [Materials Project U-value methodology](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga%2Bu-calculations/hubbard-u-values).

## Generation methods

- `alternating-index`: assigns `+ - + -` within the selected magnetic-atom list only.
- `layer`: groups coordinates along `--axis` using `--layer-tolerance` and alternates the sign per layer.
- `checkerboard`: two-colors the in-plane nearest-neighbor graph of the `--plane xy|xz|yz` plane.
- `neighbor-bipartite`: builds the PBC minimum-image nearest-neighbor graph and colors the two sublattices.
- `graph-coloring`: uses DSATUR proper coloring to build up to k sublattice candidates and maps a collinear spin to each color.
- `propagation-vector`: sets the sign from `sign(cos(2π q·r + phase))`.
- `manual-groups`: uses `--up-atoms`, `--down-atoms`, or a YAML `--group-file`.
- `manual-spins`: directly preserves signed per-atom values from `--spin-values` or `--spin-values-file`; it is intentionally excluded from candidate enumeration.
- `by-species`: splits distinct-element sublattices with `--up-species` and `--down-species`.
- `by-coordination`: splits Td/Oh sublattices by each magnetic atom's first anion-shell coordination number.
- `random`: produces reproducible random initial signs with `--seed`. This is not a physical magnetic-ordering model.

`layer` alternates the combined coordinate stack when multiple magnetic elements
are selected. If one element occurs only on the even- or odd-parity layers of
that combined stack, every atom of that element can receive the same sign even
though the other element alternates normally. The program warns when this
happens. Pass `--layer-per-species` to build and alternate an independent
coordinate stack for each element; the flag is valid only with `--method layer`
and also applies to `--layer-direction`. The default remains the combined stack.
For multi-species spinel ferrimagnets, prefer `by-coordination`; use independent
element stacks only when those layer patterns are physically intended.

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

When both `--moment` and `--moment-config` are omitted, the following generic
high-spin guesses are used as initial values and a warning lists every default that
was actually applied:

| Element | μB | Element | μB | Element | μB |
| --- | ---: | --- | ---: | --- | ---: |
| Ti | 2.0 | V | 3.0 | Cr | 3.0 |
| Mn | 5.0 | Fe | 4.0 | Co | 3.0 |
| Ni | 2.0 | Cu | 1.0 | Gd | 7.0 |

These values ignore oxidation and spin state and are only starting guesses; for
example, low-spin Co³⁺ can be approximately 0 μB. No coordination-specific defaults
are inferred. An unsupported element requires an explicit moment, and a partial
`--moment` specification remains an error rather than being silently mixed with
defaults.

Generated `DM.InitSpin` rows include inline element comments by default. With
`by-coordination`, CN=4/6 rows are also marked Td/Oh from their coordination number.
Use `--no-site-comments` for compatibility with legacy post-processing scripts; it
does not change any spin value.

For `by-species`, the union of up/down must exactly match `--magnetic-species`. That method suits distinct-element sublattices such as Ni/Co, but it cannot separate an inverse spinel in which one element occupies both Td and Oh sites — use `by-coordination` there. `by-coordination` auto-detects the anion when exactly one of O, S, Se, Te, N, F, Cl is present, and requires `--anion-species` when several candidates exist. It counts distinct periodic images of the same basis anion as separate neighbors, and the default classification is up CN=6, down CN=4. Use `--anion-cutoff` and `--coordination-tolerance` to tune the decision.

In the inverse-spinel command above, the default coordination sublattices produce
Ni(Oh)=+2 μB, Co(Td)=−2 μB, and low-spin Co(Oh)=0. Using one `Co=value`
for both CN=4 and CN=6 sites is allowed but now emits a warning because it cannot
represent those two Co sublattices independently. The Co(Oh)=0 choice is
consistent with the literature assignment of octahedral Co³⁺ as low-spin and
diamagnetic.

The experimental basis is Zhu et al.,
[“Electronic structure and magnetic properties of spinel NiCo2O4 epitaxial thin
films,” *Scientific Reports* **5**, 15201 (2015)](https://doi.org/10.1038/srep15201).
Their XAS/XMCD study describes ferrimagnetic inverse-spinel NiCo₂O₄ (`Fd-3m`,
Curie temperature 673 K) with high-spin Co²⁺/Co³⁺, but no Ni, on tetrahedral
Td(A) sites and high-spin Ni²⁺/Ni³⁺ mixed with low-spin, diamagnetic Co³⁺
(`S=0`) on octahedral Oh(B) sites. Its magnetic and transport behavior reflects
competition between antiferromagnetic super-exchange and ferromagnetic,
conducting double-exchange as the growth-dependent Ni³⁺ concentration changes,
rather than a simple two-sublattice AFM picture.

`by-coordination` separates sublattices only by coordination number (Td/Oh). It
cannot distinguish oxidation or spin states mixed within the same coordination
site, such as Ni²⁺ versus Ni³⁺ or Co³⁺ HS versus LS. To represent known
oxidation-state-specific sites, assign atom-by-atom moments with
`--site-moment-file`, or group the atoms by oxidation state in the structure and
treat those groups separately.

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

## Cell-free molecular XYZ workflows

Molecular XYZ inputs, including `examples/Fe_CO5_homogeneous_catalyst.xyz`, are
read as nonperiodic structures with no cell. Do not pass `--slab` or
`--periodic-axes` unless the input actually contains a meaningful nonsingular
cell. `analyze`, `validate`, `plot`, the GUI, and Cartesian generation methods
work directly with the XYZ atom order. This includes `alternating-index`,
`random`, Cartesian `layer`, `checkerboard`, neighbor graph methods,
`graph-coloring`, `manual-groups`, `by-species`, `by-coordination`, and
`manual-spins` when their usual method-specific inputs are valid.

Cell-free XYZ cannot use methods that explicitly request fractional
coordinates: fractional `propagation-vector` (including A/C/G presets) and
`layer --fractional-layers` report a nonsingular-cell error. A propagation
vector can be interpreted in Cartesian coordinates with
`--cartesian-coordinates`, but only use that form when it matches the intended
molecular model. Crystallographic symmetry deduplication likewise belongs to
structures with a cell.

For a small complex, `manual-spins` assigns signed moments directly to
one-based atom indices:

```bash
siesta-afm generate examples/Fe_CO5_homogeneous_catalyst.xyz \
  --magnetic-species Fe --method manual-spins \
  --spin-values 1=+2.0 --output fe_co5_spin.fdf
```

The value above is an input-format demonstration, not a claim about the
ground-state spin of Fe(CO)5. For several magnetic centers, use syntax such as
`--spin-values 1=+4.0 7=-2.0 9=+1.0`. Every selected magnetic atom must be
listed unless `--fill-unspecified-zero` is passed. A CSV alternative uses the
distinct signed schema below and is selected with `--spin-values-file`:

```csv
atom_index,spin
1,+4.0
7,-2.0
9,+1.0
```

`--spin-values` and `--spin-values-file` are mutually exclusive, and an index
whose element is not selected by `--magnetic-species` is an error. This signed
path is deliberately separate from `--site-moment-file`: the established site
moment file supplies magnitudes, so a negative `moment` there still has its
sign ignored and the ordering method determines the sign.

## Frustrated / non-bipartite graphs

If the `neighbor-bipartite` graph is not bipartite, the program does not fabricate an arbitrary two-sublattice result — it exits with an error. In that case, consider layer, propagation vector, manual groups, or a different cutoff.

`--allow-frustrated` explicitly permits an iterative Max-Cut heuristic that increases the number of opposite-sign edges. Such a result carries the following scientific warning:

> The generated spin assignment is a heuristic initial state for a frustrated magnetic network. It is not guaranteed to represent the experimental magnetic ground state.

When the graph splits into two or more connected components, only the alternating sign inside each component is determined by the graph. The relative sign between components is a deterministic convention based on the lowest atom index and has no physical meaning. In that case the program warns with the component count and sizes and suggests tuning `--neighbor-cutoff` to include interlayer superexchange or considering the `layer` / `propagation-vector` methods. A layer slab with an odd number of magnetic layers along a nonperiodic direction is not an error but an informational warning that it is an uncompensated AFM slab.

`graph-coloring` is a multi-sublattice initial-candidate generator that applies DSATUR to a non-bipartite graph. `--max-colors` defaults to 4, and you can assign per-color signs with `--color-spins "+1,-1,0"` or use `--balance-colors` to pick the color-sign permutation that minimizes the absolute sum of the actual initial moments resolved from `--moment`, per-element moments, or a site-moment file. A proper coloring only avoids equal colors on adjacent atoms; it does not minimize energy. If a collinear energy candidate for a frustrated lattice is the goal, `--allow-frustrated` max-cut is more appropriate. In `enumerate`, the color-spin permutation is varied by the attempt seed to diversify candidates.

For `generate` and `make-input`, `--spin-mode non-collinear` maps each graph
color to an in-plane direction with `theta=90°` and
`phi=360° * color / n_colors`; a three-color graph therefore gives
0°/120°/240° order. This mode is available only with `--method graph-coloring`
and cannot be combined with `--color-spins`. The default remains byte-for-byte
compatible collinear output.

## Analysis and validation

`analyze` reports magnetic-atom distance shells, the automatic cutoff, graph size / connectivity / bipartiteness, and the number of layers, and supports `--json analysis.json`.

`validate` checks for duplicate / out-of-range indices, nonzero spin on atoms that are not a selected magnetic element, up/down counts, and the net spin. When `--structure` is given, it also computes the nearest-neighbor antiparallel fraction (`AFM score`), connected components, and the per-layer distribution.

## Comparing magnetic states by total energy

Use `make-input` + `enumerate` + `prepare-array` + `collect-results` to compare
initial magnetic states while keeping the calculation settings fixed:

1. Run `make-input` once to create a complete base input containing the
   structure, basis, pseudopotential declarations, DFT+U, k-grid, and SCF
   settings. Its initial spin pattern can be arbitrary because it will be
   replaced.
2. Run `enumerate --methods ...` to create distinct spin-pattern candidates.
   The output FDF fragments contain the candidate `DM.InitSpin` blocks.
3. Pass the complete base input and candidate directory to `prepare-array`.
   Internally, `patch_fdf_text` changes only the `Spin polarized` control and
   `DM.InitSpin` block in each job's copy of the base input. Structure and all
   other calculation settings therefore remain identical across the array.
4. Run SIESTA from each generated `RUN.fdf` on the cluster; launching SIESTA is
   outside this tool's scope.
5. Run `collect-results` to write `results.csv`. Among rows with
   `scf_converged=True`, the state with the lowest `total_energy` is the leading
   candidate. When several energies are close—as is common for competing
   metastable NiCo₂O₄ states—report the complete ranking and final magnetic
   diagnostics instead of selecting one arbitrarily.

This repository's Co₃O₄ COD fixture gives an executable workflow example. The
uniform values used by `enumerate` keep moment magnitudes defined for both
methods; the coordination-specific spin in `base_input.fdf` is only a placeholder
and is replaced in every job.

```bash
siesta-afm make-input examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co --method by-coordination \
  --moment Co@4=3.0 Co@6=0.0 --output base_input.fdf

siesta-afm enumerate examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co --moment Co=3.0 Co@4=3.0 Co@6=3.0 \
  --methods by-coordination,frustrated --n-configs 2 \
  --output-dir afm_candidates

siesta-afm prepare-array base_input.fdf \
  --configs afm_candidates --output-dir siesta_jobs

# ... run each siesta_jobs/*/RUN.fdf with SIESTA on the cluster ...

siesta-afm collect-results siesta_jobs
```

`enumerate` is appropriate for topologically different arrangements such as
`--methods by-coordination,layer,frustrated`. It varies method and attempt seed;
it does **not** sweep moment magnitudes or Hubbard U. It removes identical
patterns and, by default, identifies a global up/down inversion as the same
candidate. Such an inversion has the same energy in a collinear calculation by
time-reversal symmetry, so it normally should not be run separately;
`--keep-global-spin-inversion` is available only when both conventions are
specifically needed.

For a fully periodic structure, `enumerate --symmetry-dedup` also identifies
patterns related by crystallographic symmetry. This opt-in mode requires the
`symmetry` extra (spglib); `--symprec` controls its Cartesian matching tolerance
and defaults to `1e-3` angstrom. Without `--symmetry-dedup`, enumeration and its
existing exact/global-inversion duplicate handling are unchanged.

If oxidation and spin states are themselves uncertain, build separate spin
files—for example, call `generate` or `make-input` once with Co(Oh)=0 (LS) and
once with nonzero Co(Oh) (HS). Copy those files, plus any enumerated candidates,
into one configuration directory; give every file a unique `config_id` and file
name; and merge or write `manifest.csv` with this exact schema:

```csv
config_id,method,n_up,n_down,net_spin,afm_score,file
```

Each row describes one spin file; use labels such as `by-coordination-ls` and
`by-coordination-hs` in `method`, and preserve or recompute the counts, net spin,
and AFM score for that file. `prepare-array` can then patch all of them into the
same base input. A Hubbard-U sweep is different: because U belongs to the base
calculation settings, make a separate base input and job array for each U. For
per-atom oxidation-state assignments, `--site-moment-file` is safer than relying
on coordination alone.

`collect-results` reads energy, final net spin, per-atom Mulliken/Hirshfeld spin, sign-retention fraction, collapse, and convergence markers from the common SIESTA `.out`/`.log` representations in each folder and writes `results.csv`. Different SIESTA versions may use different output wording, which can require extending the regular expressions.

### The same workflow in the GUI

The `Batch workflow` tab is an optional tool for comparing multiple magnetic
initial states by total energy; it is not required to generate a single SIESTA
input. It exposes the candidate, job-preparation, and result stages described
above without changing their file formats. In `Candidates`,
choose one or more methods and an output directory; the current magnetization
table supplies `--magnetic-species` and `--moment`, and the generated rows plus
all skipped-method warnings are shown in the tab. When using `manual-groups`,
select its existing group file in the same panel. For a fully periodic input,
the default-off `Symmetry-aware dedup` checkbox enables the same spglib-backed
candidate collapsing as the CLI. `Prepare jobs` requires an
explicit, already saved complete FDF from `Build complete SIESTA input
(make-input)...`, a candidate directory containing `manifest.csv`, and an
output directory. It shows the resulting `folders.list` entries.

After running SIESTA outside this tool, use `Results` either to collect a jobs
directory or to load an existing `results.csv` without collecting again. Rows
are sorted by ascending `total_energy`; unconverged rows are gray, and converged
states within 0.01 eV of the lowest energy are highlighted as a group. The GUI
does not automatically choose one of several close candidates.

Drag the horizontal divider between the 3D preview and the results notebook to
give `Batch workflow` more vertical space. Each of its three subtabs also has a
vertical scrollbar and supports the mouse wheel; when the pointer is over a
candidate or results table, the wheel scrolls that table instead of the outer
tab.

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

Opening a CIF, FDF, XYZ, POSCAR/CONTCAR, or XV file immediately lists every element in
the magnetization table. Elements covered by the built-in moment table are checked by
default; anions and unsupported elements remain unchecked. The table is the source of
truth for `use | element | label | CN | value (μB) | count | role`: double-click a use
cell to toggle an element, or double-click label, value, and the `by-species` role to
edit them. Element, CN, and count are read-only. The equivalent
`--magnetic-species ... --moment ...` CLI options are shown immediately below the
table for reuse in scripts. A site-moment CSV remains available for atom-specific
overrides and has the same priority as the CLI: site CSV > table (`Element@CN` or
`Element`) value.

With `by-coordination`, the GUI creates one row per `(Element, CN)` group and derives
the displayed geometry from ligand vectors, not from CN alone. It distinguishes, for
example, square-planar Cu(CN=4) in CuO from tetrahedral Co(CN=4) in a spinel by
counting ligand pairs with angles of at least 170°. Geometry labels are editable
estimates; moment syntax remains `Element@CN`. An edited label is also written to the
corresponding `DM.InitSpin` comments. If coordination analysis fails, the GUI keeps
usable element-level rows and reports the reason instead of clearing the table.
Each `(Element, CN)` use checkbox is independent. Unchecking, for example, only
`Co@6` leaves `Co@4` enabled and keeps Co in `--magnetic-species`; the omitted
`Co@6` moment then produces the same explicit partial-moment error as the CLI.
Set that row's moment to `0.0`, or use `--exclude-atoms` / `--adsorbate-indices`
for specific atom indices, when those sites must be fully excluded.

When `manual-spins` is selected, structures with at most 60 atoms show a second
atom-order table (`atom index | element | spin`) whose signed spin cells are
edited by double-clicking. Edits use the normal debounced live-preview path and
preserve the current 3D camera. For larger structures the table is replaced by
an `atom_index,spin` CSV selector to avoid hundreds of GUI rows.

Only settings relevant to the selected method are displayed. The input and result
areas are separated by a draggable pane, and short labels plus help text keep input
widgets readable at the default window size. Parameter changes still use the 400 ms
live-preview debounce. `Include element/CN comments in DM.InitSpin` controls the same
output feature as CLI `--no-site-comments`; default-moment and spin-state warnings
appear in both the status bar and `Analysis` tab. Existing spin files can also be
opened on top of the current structure. The preview shows atom indices automatically
for structures with at most 60 atoms and hides them for larger structures; `Show atom
indices` can override that choice for the currently loaded structure. The preview also
provides one spin-visibility checkbox per element plus `Show bonds` and a bond-radius
scale. After generation or opening a spin file, each element checkbox shows its assigned
up, down, and zero/unassigned site counts. These counts stay unchanged when the checkbox
is toggled; unchecking an element hides only its spin color and arrow, not its atom
marker. A `by-coordination` result also adds one checkbox per detected CN; the element
and CN filters are combined, so a spin is visible only when both its element and CN are
checked. CN controls are omitted for other methods and results without stored CN data.
Bond display is enabled by default only for structures with at most 60 atoms.

After generation, the `Sites` tab lists every magnetic atom in input order with its
element, CN, sublattice, sign, and moment. Its footer shows `n_up`, `n_down`,
`n_zero`, and the net initial moment. The spin-file viewer populates the same table,
with CN and sublattice shown as `-` because those values are not stored in a spin
block. Rows start in input atom order; click the `CN` column header to group them by
ascending CN, then click it again to restore atom-index order.

The analysis corresponding to the CLI `analyze` is not a separate button — it runs automatically on generation and live updates and appears in the right-hand `Analysis` tab as distance shells, cutoff, connectivity, bipartiteness, and the number of layers.

Export can save the DM.InitSpin block, a complete SIESTA starting input, a patched
SIESTA input that never overwrites the original, and an XYZ/CIF structure with the
initial magmom included. The complete-input action displays the same convergence
warning and uses the same renderer as `make-input`. The CLI is the reference
scientific implementation, and the GUI controller uses the same core functions.
For a complete runnable starting FDF, use the prominent
`Build complete SIESTA input (make-input)...` button in the fixed `Primary actions`
bar; the similarly named Export action remains available as a secondary entry point.

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

The small structures in `examples/` are for confirming CLI behavior and do not substitute for a converged calculation model. `Fe_CO5_homogeneous_catalyst.xyz` is an idealized trigonal-bipyramidal molecular geometry for exercising the nonperiodic workflow. In particular, `NiCo2O4_311_slab.cif` is a multi-element input/output demo, not a real spinel structure with constructed Td/Oh coordination, so it must not be used as a `by-coordination` validation model; its Ni/Co-O distances intentionally exceed typical covalent-bond cutoffs, so `--show-bonds`/`Show bonds` will find few or no bonds on it. To try the element filter and bond lines on a physically realistic multi-element structure, use `examples/NiCo2O4_spinel_demo.cif` instead (the real Co3O4 spinel geometry from COD 1538531 with its eight tetrahedral Co sites relabeled Ni).

## Tests

```bash
python -m pytest
```

The tests cover FDF/include/patch, 1D / square / triangle / disconnected graphs, slab PBC, layer clustering, ordering, the one-based writer, and validation error detection.

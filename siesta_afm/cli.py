"""Command-line interface for siesta-afm."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .fdf_writer import patch_fdf_file, patch_fdf_text, render_dm_init_spin
from .io import parse_dm_init_spin, read_structure
from .magnetic_sites import (
    guess_oxidation_states,
    load_moment_config,
    select_magnetic_sites,
)
from .neighbors import build_neighbor_graph
from .ordering import NonBipartiteError
from .results import collect_results, prepare_array
from .validation import (
    analyze_structure,
    format_analysis,
    format_validation,
    validate_spin_file,
)
from .visualize import plot_spin_pattern
from .workflows import generate_assignment


METHODS = [
    "alternating-index",
    "random",
    "layer",
    "checkerboard",
    "neighbor-bipartite",
    "graph-coloring",
    "propagation-vector",
    "manual-groups",
    "by-species",
    "by-coordination",
]


def _configure_windows_stdio() -> None:
    """Allow scientific Unicode labels in legacy Windows console locales."""

    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_input(
    parser: argparse.ArgumentParser, *, help_text: str = "structure file"
) -> None:
    parser.add_argument("input_path", nargs="?", help=help_text)
    parser.add_argument("--input", dest="input_option", help=help_text)


def _input_path(args: argparse.Namespace) -> str:
    value = getattr(args, "input_option", None) or getattr(args, "input_path", None)
    if not value:
        raise ValueError("an input path is required (positional or --input)")
    return str(value)


def _add_structure_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slab", action="store_true", help="use xy PBC and nonperiodic z"
    )
    parser.add_argument(
        "--periodic-axes",
        choices=["x", "y", "z", "xy", "xz", "yz", "xyz"],
        help="override periodic axes",
    )


def _add_site_controls(
    parser: argparse.ArgumentParser, *, require_moment: bool = True
) -> None:
    parser.add_argument("--magnetic-species", nargs="+", required=True)
    if require_moment:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--moment",
            nargs="+",
            help="global magnitude, Element=value, or Element@CN=value specifications",
        )
        group.add_argument(
            "--moment-config", help="YAML file containing a moments mapping"
        )
        parser.add_argument("--site-moment-file")
        parser.add_argument("--guess-oxidation-states", action="store_true")
    parser.add_argument("--exclude-atoms")
    parser.add_argument("--adsorbate-indices")


def _add_neighbor_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cutoff", "--neighbor-cutoff", default="auto")
    parser.add_argument("--neighbor-shell", type=int, default=1)


def _add_site_comment_control(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-site-comments",
        dest="site_comments",
        action="store_false",
        default=True,
        help="omit element/CN comments from DM.InitSpin rows",
    )


def _add_ordering_controls(parser: argparse.ArgumentParser) -> None:
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--axis", choices=list("xyz"), default="z")
    layer_group.add_argument(
        "--layer-direction",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Cartesian layer-normal direction, e.g. 1 1 1 for NiO AFM-II",
    )
    parser.add_argument(
        "--layer-tolerance",
        type=float,
        default=0.25,
        help="layer clustering tolerance in Å (fractional units with --fractional-layers)",
    )
    parser.add_argument(
        "--layer-pattern", choices=["alternating"], default="alternating"
    )
    parser.add_argument("--fractional-layers", action="store_true")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    _add_neighbor_controls(parser)
    parser.add_argument("--allow-frustrated", action="store_true")
    q_group = parser.add_mutually_exclusive_group()
    q_group.add_argument(
        "--q-vector",
        type=float,
        nargs=3,
        help=(
            "propagation vector in fractional coordinates of the input cell; "
            "scale it when using a supercell"
        ),
    )
    q_group.add_argument(
        "--afm-type",
        choices=["A", "C", "G"],
        help="propagation-vector preset: A=(0,0,1/2), C=(1/2,1/2,0), G=(1/2,1/2,1/2)",
    )
    parser.add_argument("--phase", type=float, default=0.0)
    coordinate_group = parser.add_mutually_exclusive_group()
    coordinate_group.add_argument(
        "--fractional-coordinates",
        dest="fractional_coordinates",
        action="store_true",
        default=True,
    )
    coordinate_group.add_argument(
        "--cartesian-coordinates",
        dest="fractional_coordinates",
        action="store_false",
    )
    parser.add_argument("--up-atoms")
    parser.add_argument("--down-atoms")
    parser.add_argument("--group-file")
    parser.add_argument(
        "--up-species",
        nargs="+",
        help=(
            "up sublattice for by-species; inverse spinels with one element on "
            "Td/Oh sites require by-coordination"
        ),
    )
    parser.add_argument("--down-species", nargs="+")
    parser.add_argument("--anion-species", nargs="+")
    parser.add_argument("--anion-cutoff", default="auto")
    parser.add_argument("--up-coordination", type=int, nargs="+", default=[6])
    parser.add_argument("--down-coordination", type=int, nargs="+", default=[4])
    parser.add_argument("--coordination-tolerance", type=int, default=0)
    parser.add_argument("--max-colors", type=int, default=4)
    parser.add_argument(
        "--color-spins",
        help='comma-separated color mapping, e.g. "+1,-1,0"',
    )
    parser.add_argument(
        "--balance-colors",
        action="store_true",
        help="permute the color-spin map to minimize absolute net initial moment",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siesta-afm",
        description="Generate SIESTA AFM DM.InitSpin initial states without reordering atoms.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate an AFM initial-spin block"
    )
    _add_input(generate)
    _add_structure_controls(generate)
    _add_site_controls(generate)
    generate.add_argument("--method", choices=METHODS, required=True)
    _add_ordering_controls(generate)
    generate.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed or graph-coloring permutation index (default: 0)",
    )
    generate.add_argument("--output")
    generate.add_argument("--write-zero-spins", action="store_true")
    _add_site_comment_control(generate)
    generate.add_argument("--patch-input", action="store_true")
    generate.add_argument("--in-place", action="store_true")
    generate.add_argument("--backup", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    analyze = subparsers.add_parser("analyze", help="analyze the magnetic graph")
    _add_input(analyze)
    _add_structure_controls(analyze)
    _add_site_controls(analyze, require_moment=False)
    _add_neighbor_controls(analyze)
    analyze_layer_group = analyze.add_mutually_exclusive_group()
    analyze_layer_group.add_argument("--axis", choices=list("xyz"), default="z")
    analyze_layer_group.add_argument("--layer-direction", type=float, nargs=3)
    analyze.add_argument(
        "--layer-tolerance",
        type=float,
        default=0.25,
        help="layer tolerance in Å (fractional units with --fractional-layers)",
    )
    analyze.add_argument("--fractional-layers", action="store_true")
    analyze.add_argument("--json", dest="json_output")
    analyze.set_defaults(func=_cmd_analyze)

    validate = subparsers.add_parser("validate", help="validate a DM.InitSpin block")
    validate.add_argument("spin_file")
    validate.add_argument("--structure")
    validate.add_argument("--magnetic-species", nargs="+")
    validate.add_argument("--cutoff", "--neighbor-cutoff", default="auto")
    _add_structure_controls(validate)
    validate.add_argument("--json", dest="json_output")
    validate.set_defaults(func=_cmd_validate)

    patch = subparsers.add_parser("patch", help="patch a spin block into an FDF input")
    patch.add_argument("input_fdf")
    patch.add_argument("--spin-file", required=True)
    patch.add_argument("--output")
    patch.add_argument("--in-place", action="store_true")
    patch.add_argument("--backup", action="store_true")
    patch.set_defaults(func=_cmd_patch)

    plot = subparsers.add_parser("plot", help="visualize or export a spin pattern")
    _add_input(plot)
    _add_structure_controls(plot)
    plot.add_argument("--spin-file", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--show-indices", action="store_true")
    plot.add_argument("--color-by-layer", action="store_true")
    plot.add_argument(
        "--color-mode",
        choices=["sign", "value"],
        default="sign",
        help="color atoms by spin sign (default) or continuous spin value",
    )
    plot.add_argument(
        "--up-color",
        help="spin-up color in sign mode only (default: tab:red)",
    )
    plot.add_argument(
        "--down-color",
        help="spin-down color in sign mode only (default: tab:blue)",
    )
    plot.add_argument("--nonmagnetic-color", default="0.65")
    plot.set_defaults(func=_cmd_plot)

    enumerate_parser = subparsers.add_parser(
        "enumerate", help="generate distinct AFM candidate configurations"
    )
    _add_input(enumerate_parser)
    _add_structure_controls(enumerate_parser)
    _add_site_controls(enumerate_parser)
    _add_ordering_controls(enumerate_parser)
    enumerate_parser.add_argument("--methods", required=True)
    enumerate_parser.add_argument("--n-configs", type=int, default=8)
    enumerate_parser.add_argument("--output-dir", required=True)
    enumerate_parser.add_argument("--keep-global-spin-inversion", action="store_true")
    _add_site_comment_control(enumerate_parser)
    enumerate_parser.set_defaults(func=_cmd_enumerate)

    array = subparsers.add_parser(
        "prepare-array", help="create one SIESTA calculation folder per configuration"
    )
    array.add_argument("input_fdf")
    array.add_argument("--configs", required=True)
    array.add_argument("--template")
    array.add_argument("--output-dir", required=True)
    array.set_defaults(func=_cmd_prepare_array)

    collect = subparsers.add_parser(
        "collect-results", help="collect energies and final magnetic states"
    )
    collect.add_argument("jobs_dir")
    collect.add_argument("--output")
    collect.add_argument("--collapse-initial", type=float, default=0.5)
    collect.add_argument("--collapse-final", type=float, default=0.1)
    collect.set_defaults(func=_cmd_collect_results)
    return parser


def _read_from_args(args: argparse.Namespace):
    return read_structure(
        _input_path(args), slab=args.slab, periodic_axes=args.periodic_axes
    )


def _workflow_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "exclude_atoms": args.exclude_atoms,
        "adsorbate_indices": args.adsorbate_indices,
        "site_moment_file": args.site_moment_file,
        "axis": args.axis,
        "layer_direction": args.layer_direction,
        "layer_tolerance": args.layer_tolerance,
        "fractional_layers": args.fractional_layers,
        "plane": args.plane,
        "cutoff": args.cutoff,
        "neighbor_shell": args.neighbor_shell,
        "allow_frustrated": args.allow_frustrated,
        "q_vector": args.q_vector,
        "afm_type": args.afm_type,
        "phase": args.phase,
        "fractional_coordinates": args.fractional_coordinates,
        "up_atoms": args.up_atoms,
        "down_atoms": args.down_atoms,
        "group_file": args.group_file,
        "up_species": args.up_species,
        "down_species": args.down_species,
        "anion_species": args.anion_species,
        "anion_cutoff": args.anion_cutoff,
        "up_coordination": args.up_coordination,
        "down_coordination": args.down_coordination,
        "coordination_tolerance": args.coordination_tolerance,
        "max_colors": args.max_colors,
        "color_spins": args.color_spins,
        "balance_colors": args.balance_colors,
    }


def _moment_values(args: argparse.Namespace) -> list[str] | None:
    if args.moment is not None:
        return list(args.moment)
    if args.moment_config is not None:
        return load_moment_config(args.moment_config)
    return None


def _cmd_generate(args: argparse.Namespace) -> int:
    input_path = _input_path(args)
    structure = _read_from_args(args)
    if args.guess_oxidation_states:
        states = guess_oxidation_states(structure)
        print(
            "WARNING: oxidation states were guessed heuristically and must be "
            "checked before use.",
            file=sys.stderr,
        )
        print(
            "Guessed oxidation states: " + " ".join(f"{v:g}" for v in states),
            file=sys.stderr,
        )
    _, assignment, spins = generate_assignment(
        structure,
        args.magnetic_species,
        args.method,
        _moment_values(args),
        seed=args.seed,
        **_workflow_kwargs(args),
    )
    for warning in assignment.warnings:
        print(f"WARNING:\n{warning}", file=sys.stderr)
    spin_text = render_dm_init_spin(
        spins,
        method=assignment.method,
        magnetic_species=args.magnetic_species,
        metadata=assignment.metadata,
        structure=structure,
        write_zero_spins=args.write_zero_spins,
        site_comments=args.site_comments,
    )
    if not args.patch_input:
        if args.in_place or args.backup:
            raise ValueError("--in-place/--backup require --patch-input")
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(spin_text, encoding="utf-8")
            print(destination)
        else:
            print(spin_text, end="")
        return 0

    if Path(input_path).suffix.lower() != ".fdf":
        raise ValueError("--patch-input requires an FDF input structure")
    source = Path(input_path)
    if (
        args.in_place
        and args.output is not None
        and Path(args.output).resolve() != source.resolve()
    ):
        raise ValueError("--in-place cannot be combined with a different --output")
    if args.in_place:
        destination = source
    else:
        destination = (
            Path(args.output)
            if args.output
            else source.with_name(source.stem + "_afm.fdf")
        )
        if destination.resolve() == source.resolve():
            raise ValueError("refusing to overwrite input without --in-place")
    if args.backup:
        shutil.copy2(source, Path(str(source) + ".bak"))
    patched = patch_fdf_text(source.read_text(encoding="utf-8-sig"), spin_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(patched, encoding="utf-8")
    print(destination)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    structure = _read_from_args(args)
    indices = select_magnetic_sites(
        structure,
        args.magnetic_species,
        exclude_atoms=args.exclude_atoms,
        adsorbate_indices=args.adsorbate_indices,
    )
    report = analyze_structure(
        structure,
        indices,
        magnetic_species=args.magnetic_species,
        cutoff=args.cutoff,
        neighbor_shell=args.neighbor_shell,
        axis=args.axis,
        layer_tolerance=args.layer_tolerance,
        fractional_layers=args.fractional_layers,
        layer_direction=args.layer_direction,
    )
    print(format_analysis(report))
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    structure = (
        read_structure(args.structure, slab=args.slab, periodic_axes=args.periodic_axes)
        if args.structure
        else None
    )
    report = validate_spin_file(
        args.spin_file,
        structure=structure,
        magnetic_species=args.magnetic_species,
        cutoff=args.cutoff,
    )
    print(format_validation(report))
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0 if report.valid else 1


def _cmd_patch(args: argparse.Namespace) -> int:
    destination = patch_fdf_file(
        args.input_fdf,
        args.spin_file,
        output_path=args.output,
        in_place=args.in_place,
        backup=args.backup,
    )
    print(destination)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    structure = _read_from_args(args)
    spins = {index - 1: value for index, value in parse_dm_init_spin(args.spin_file)}
    if args.color_mode == "value" and (
        args.up_color is not None or args.down_color is not None
    ):
        print(
            "WARNING: --up-color/--down-color are ignored in value color mode",
            file=sys.stderr,
        )
    destination = plot_spin_pattern(
        structure,
        spins,
        args.output,
        show_indices=args.show_indices,
        color_by_layer=args.color_by_layer,
        color_mode=args.color_mode,
        up_color=args.up_color or "tab:red",
        down_color=args.down_color or "tab:blue",
        nonmagnetic_color=args.nonmagnetic_color,
    )
    print(destination)
    return 0


def _canonical_pattern(signs: dict[int, int], keep_inversion: bool) -> tuple[int, ...]:
    pattern = tuple(signs[index] for index in sorted(signs))
    if keep_inversion:
        return pattern
    inverse = tuple(-value for value in pattern)
    return min(pattern, inverse)


def _cmd_enumerate(args: argparse.Namespace) -> int:
    if args.n_configs <= 0:
        raise ValueError("--n-configs must be positive")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method")
    allowed = set(METHODS) | {"frustrated"}
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"unsupported enumeration method: {unknown[0]}")
    structure = _read_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[int, ...]] = set()
    manifest: list[dict[str, object]] = []
    failures: list[str] = []
    notices: list[str] = []
    max_attempts = max(args.n_configs * 20, len(methods))
    for attempt in range(max_attempts):
        if len(manifest) >= args.n_configs:
            break
        method = methods[attempt % len(methods)]
        try:
            indices, assignment, spins = generate_assignment(
                structure,
                args.magnetic_species,
                method,
                _moment_values(args),
                seed=attempt,
                **_workflow_kwargs(args),
            )
        except (ValueError, NonBipartiteError) as exc:
            message = f"{method}: {exc}"
            if message not in failures:
                failures.append(message)
            continue
        key = _canonical_pattern(assignment.signs, args.keep_global_spin_inversion)
        if key in seen:
            continue
        try:
            graph, _, _ = build_neighbor_graph(
                structure, indices, args.cutoff, neighbor_shell=args.neighbor_shell
            )
        except ValueError as exc:
            message = f"{method}: {exc}"
            if message not in failures:
                failures.append(message)
            continue
        score = (
            sum(
                assignment.signs[left] * assignment.signs[right] < 0
                for left, right in graph.edges
            )
            / graph.number_of_edges()
            if graph.number_of_edges()
            else 0.0
        )
        seen.add(key)
        config_id = f"{len(manifest) + 1:03d}"
        file_name = f"afm_{config_id}.fdf"
        text = render_dm_init_spin(
            spins,
            method=assignment.method,
            magnetic_species=args.magnetic_species,
            metadata=assignment.metadata,
            structure=structure,
            site_comments=args.site_comments,
        )
        (output_dir / file_name).write_text(text, encoding="utf-8")
        manifest.append(
            {
                "config_id": config_id,
                "method": assignment.method,
                "n_up": assignment.n_up,
                "n_down": assignment.n_down,
                "net_spin": sum(spins.values()),
                "afm_score": score,
                "file": file_name,
            }
        )
        for warning in assignment.warnings:
            if warning not in notices:
                notices.append(warning)
    if not manifest:
        detail = "\n".join(failures) if failures else "no distinct patterns"
        raise ValueError(f"no AFM configurations could be generated:\n{detail}")
    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    for warning in notices:
        print(f"WARNING:\n{warning}", file=sys.stderr)
    print(f"Generated {len(manifest)} distinct configuration(s) in {output_dir}")
    for failure in failures:
        print(f"WARNING: skipped {failure}", file=sys.stderr)
    if len(manifest) < args.n_configs:
        print(
            f"WARNING: requested {args.n_configs}, but only {len(manifest)} distinct "
            "patterns were found.",
            file=sys.stderr,
        )
    return 0


def _cmd_prepare_array(args: argparse.Namespace) -> int:
    folders = prepare_array(
        args.input_fdf,
        args.configs,
        args.output_dir,
        template=args.template,
    )
    print(f"Prepared {len(folders)} job folder(s) in {args.output_dir}")
    return 0


def _cmd_collect_results(args: argparse.Namespace) -> int:
    rows = collect_results(
        args.jobs_dir,
        output_csv=args.output,
        collapse_initial=args.collapse_initial,
        collapse_final=args.collapse_final,
    )
    print(f"Collected {len(rows)} job(s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_windows_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NonBipartiteError as exc:
        print(f"ERROR:\n{exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

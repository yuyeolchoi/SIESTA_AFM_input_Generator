"""Optional Streamlit interface for the siesta-afm workflow."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


_GUI_INSTALL_HINT = 'python -m pip install -e ".[gui]"'


def _missing_gui_dependency(exc: ImportError) -> RuntimeError:
    return RuntimeError(
        "the GUI requires optional dependencies; install them with "
        f"'{_GUI_INSTALL_HINT}'"
    )


def _running_in_streamlit() -> bool:
    """Return whether this module is being executed by a Streamlit runtime."""

    try:
        import streamlit as st
    except ImportError:
        return False
    try:
        return bool(st.runtime.exists())
    except AttributeError:
        return False


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise _missing_gui_dependency(exc) from exc

    # Streamlit executes this file as a script, without package-relative import
    # context, after the launcher bootstraps its runtime.
    from siesta_afm.fdf_writer import render_dm_init_spin
    from siesta_afm.io import read_structure
    from siesta_afm.validation import analyze_structure
    from siesta_afm.visualize import plot_spin_pattern
    from siesta_afm.workflows import generate_assignment

    st.set_page_config(page_title="SIESTA AFM", layout="wide")
    st.title("SIESTA AFM initial-spin generator")
    st.warning(
        "Generated AFM patterns are initial magnetic guesses. They do not guarantee "
        "the magnetic ground state. For frustrated or non-bipartite lattices, compare "
        "multiple initial spin configurations."
    )

    uploaded = st.file_uploader(
        "Structure file", type=["cif", "xyz", "fdf", "xv", "vasp"]
    )
    left, right = st.columns(2)
    with left:
        species_text = st.text_input("Magnetic species", "Cu")
        method = st.selectbox(
            "Method",
            [
                "layer",
                "neighbor-bipartite",
                "alternating-index",
                "checkerboard",
                "propagation-vector",
            ],
        )
        moment = st.number_input("Moment magnitude", min_value=0.0, value=0.5, step=0.1)
        axis = st.selectbox("Layer axis", ["z", "x", "y"])
    with right:
        auto_cutoff = st.checkbox("Automatic first-shell cutoff", value=True)
        cutoff_value = st.slider("Neighbor cutoff (Å)", 0.5, 10.0, 3.2, 0.05)
        cutoff_text = "auto" if auto_cutoff else str(cutoff_value)
        tolerance = st.slider("Layer tolerance (Å)", 0.01, 1.0, 0.25, 0.01)
        slab = st.checkbox("Slab (periodic xy, nonperiodic z)", value=False)
        q_text = st.text_input("q-vector", "0.5 0.5 0.5")
        allow_frustrated = st.checkbox("Allow frustrated heuristic", value=False)
        color_mode_label = st.selectbox(
            "Color mode", ["spin sign", "spin value"], index=0
        )

    if uploaded and st.button("Generate", type="primary"):
        suffix = Path(uploaded.name).suffix or ".vasp"
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / (
                    "POSCAR" if suffix == ".vasp" else uploaded.name
                )
                path.write_bytes(uploaded.getvalue())
                structure = read_structure(path, slab=slab)
                species = species_text.split()
                q_vector = [float(value) for value in q_text.replace(",", " ").split()]
                indices, assignment, spins = generate_assignment(
                    structure,
                    species,
                    method,
                    str(moment),
                    axis=axis,
                    layer_tolerance=tolerance,
                    cutoff=cutoff_text,
                    allow_frustrated=allow_frustrated,
                    q_vector=q_vector,
                )
                block = render_dm_init_spin(
                    spins,
                    method=assignment.method,
                    magnetic_species=species,
                    metadata=assignment.metadata,
                )
                report = analyze_structure(
                    structure,
                    indices,
                    magnetic_species=species,
                    cutoff=cutoff_text,
                    axis=axis,
                    layer_tolerance=tolerance,
                )
                preview_path = Path(directory) / "spin_preview.png"
                plot_spin_pattern(
                    structure,
                    spins,
                    preview_path,
                    show_indices=True,
                    color_mode="value" if color_mode_label == "spin value" else "sign",
                )
                preview_bytes = preview_path.read_bytes()
            st.subheader("3D structure preview")
            st.image(preview_bytes, use_container_width=True)
            st.subheader("Magnetic graph")
            st.json(report)
            st.subheader("Up/down atoms")
            st.dataframe(
                [
                    {
                        "atom_index": index + 1,
                        "element": structure.symbols[index],
                        "spin": spins[index],
                    }
                    for index in sorted(spins)
                ],
                use_container_width=True,
            )
            st.subheader("DM.InitSpin preview")
            st.code(block, language="text")
            st.download_button(
                "Download FDF", block, file_name="afm_spin.fdf", mime="text/plain"
            )
            for warning in assignment.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error(str(exc))


def main() -> int:
    """Launch the installed Streamlit application."""

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise _missing_gui_dependency(exc) from exc
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    result = streamlit_cli.main()
    return int(result or 0)


def _module_entrypoint() -> int | None:
    """Bootstrap Streamlit once, then render when Streamlit reloads this file."""

    if _running_in_streamlit():
        run()
        return None
    return main()


if __name__ == "__main__":
    exit_code = _module_entrypoint()
    if exit_code is not None:
        raise SystemExit(exit_code)

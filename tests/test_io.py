from siesta_afm.io import parse_dm_init_spin


def test_dm_init_spin_angle_parsing_is_opt_in_and_normalizes_collinear_rows() -> None:
    text = (
        "%block DM.InitSpin\n"
        "1 +1.0\n"
        "2 -0.5\n"
        "3 0.75 90.0 240.0\n"
        "%endblock DM.InitSpin\n"
    )
    warnings: list[str] = []

    assert parse_dm_init_spin(text, warnings=warnings) == [
        (1, 1.0),
        (2, -0.5),
        (3, 0.75),
    ]
    assert parse_dm_init_spin(text, include_angles=False) == [
        (1, 1.0),
        (2, -0.5),
        (3, 0.75),
    ]
    assert len(warnings) == 1
    assert "theta/phi" in warnings[0]
    assert parse_dm_init_spin(text, include_angles=True) == [
        (1, 1.0, 0.0, 0.0),
        (2, 0.5, 180.0, 0.0),
        (3, 0.75, 90.0, 240.0),
    ]


def test_dm_init_spin_angle_parser_defaults_missing_phi_to_zero() -> None:
    text = "%block DM.InitSpin\n1 1.5 45\n%endblock DM.InitSpin\n"

    assert parse_dm_init_spin(text, include_angles=True) == [
        (1, 1.5, 45.0, 0.0)
    ]

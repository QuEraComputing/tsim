from tsim.utils.program_text import shorthand_to_stim, stim_to_shorthand


def test_shorthand_to_stim_t_and_t_dag():
    text = "T 0 1\nT_DAG 2"
    expected = "S[T] 0 1\nS_DAG[T] 2"
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_rotations():
    text = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2"
    expected = (
        "I[R_X(theta=0.25*pi)] 0\n" "I[R_Y(theta=-0.5*pi)] 1\n" "I[R_Z(theta=0.3*pi)] 2"
    )
    assert shorthand_to_stim(text) == expected


def test_shorthand_to_stim_u3():
    text = "U3(0.3, 0.24, 0.49) 0"
    expected = "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 0"
    assert shorthand_to_stim(text) == expected


def test_stim_to_shorthand_t_and_t_dag():
    text = "S[T] 0 1\nS_DAG[T] 2"
    expected = "T 0 1\nT_DAG 2"
    assert stim_to_shorthand(text) == expected


def test_stim_to_shorthand_rotations_and_u3():
    text = (
        "I[R_X(theta=0.25*pi)] 0\n"
        "I[R_Y(theta=-0.5*pi)] 1\n"
        "I[R_Z(theta=0.3*pi)] 2\n"
        "I[U3(theta=0.3*pi, phi=0.24*pi, lambda=0.49*pi)] 3"
    )
    expected = "R_X(0.25) 0\nR_Y(-0.5) 1\nR_Z(0.3) 2\nU3(0.3, 0.24, 0.49) 3"
    assert stim_to_shorthand(text) == expected


def test_shorthand_roundtrip():
    text = "T 0\nR_X(0.5) 1\nU3(0.1, 0.2, 0.3) 2"
    assert stim_to_shorthand(shorthand_to_stim(text)) == text

"""Checks for the cultivation feasibility case (true values: width 4, acceptance 0.687)."""

from pathlib import Path

from checks import (
    answer_has_value,
    answer_matches,
    no_stabilizer_rank,
    postselects_in_tsim,
    uses_detector_sampler,
)


def check(answer: str, transcript: str, work: Path) -> dict:
    return {
        "reports active width 4": answer_matches(
            answer, r"width\D{0,30}\b4\b", "width 4"
        ),
        "reports acceptance near 68.7%": answer_has_value(
            answer, 0.675, 0.70, "acceptance"
        ),
        "reports the thread count": answer_matches(
            answer, r"\d+\s*(threads?|cores?)", "threads"
        ),
        "uses the detector sampler": uses_detector_sampler(transcript),
        "postselects inside Tsim": postselects_in_tsim(transcript),
        "no stabilizer-rank strategy": no_stabilizer_rank(transcript),
    }

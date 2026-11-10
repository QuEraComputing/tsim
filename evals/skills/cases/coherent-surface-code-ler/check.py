"""Checks for the coherent surface-code logical error rate case (true value about 3.6e-5)."""

import re
from pathlib import Path

from checks import (
    answer_matches,
    no_sinter_collect,
    no_stabilizer_rank,
    uses_detector_sampler,
    writes_json_with,
)


def _matching_model_is_decomposed(transcript: str) -> tuple[bool, str]:
    if "pymatching" not in transcript.lower():
        return True, "PyMatching not used"
    ok = re.search(
        r"detector_error_model\([^)]*decompose_errors\s*=\s*True", transcript
    )
    return ok is not None, (
        "decompose_errors=True" if ok else "PyMatching on an undecomposed model"
    )


def check(answer: str, transcript: str, work: Path) -> dict:
    return {
        "logical error rate in the right range": answer_matches(
            answer,
            r"\b3\.[2-9]\d*[^\n]{0,25}?(e-0?5|10\^?[-\u2212]5|10\u207b\u2075)",
            "a rate near 3.6e-5",
        ),
        "reports an interval or error bar": answer_matches(
            answer, r"interval|±|\+/-|standard error|\bCI\b", "an error bar"
        ),
        "saves counts to JSON": writes_json_with(work, "errors"),
        "PyMatching model decomposed": _matching_model_is_decomposed(transcript),
        "uses the detector sampler": uses_detector_sampler(transcript),
        "no sinter.collect": no_sinter_collect(transcript),
        "no stabilizer-rank strategy": no_stabilizer_rank(transcript),
    }

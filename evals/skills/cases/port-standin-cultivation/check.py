"""Checks for the stand-in porting case (true values: widths 4 and 6, acceptance 0.687 / 0.144)."""

import re
from pathlib import Path

from checks import (
    answer_matches,
    no_stabilizer_rank,
    postselects_in_tsim,
)


def _converted_circuit(work: Path) -> tuple[bool, str]:
    for path in work.rglob("*.stim"):
        if "inputs" in path.parts:
            continue
        lines = re.findall(r"^\s*T(_DAG)?\b", path.read_text(), re.MULTILINE)
        if len(lines) >= 5:
            return True, f"{path.relative_to(work)}: {len(lines)} T/T_DAG lines"
    return False, "no converted circuit with T gates saved"


def _checked_against_stim(answer: str, transcript: str) -> tuple[bool, str]:
    # NOTE: the stim.Circuit is often built in a helper far from where it is sampled, so
    # require one in the code and a Tsim-vs-Stim comparison in the answer.
    built = re.search(r"stim\.Circuit\(", transcript)
    compared = re.search(
        r"Stim[^\n]{0,120}\b(agree|match|consistent|within)|"
        r"\b(agree|match|consistent)[^\n]{0,80}\bStim\b",
        answer,
    )
    if not built:
        return False, "no stim.Circuit built"
    return compared is not None, (
        "Tsim vs Stim compared" if compared else "no comparison with Stim"
    )


def _reports_widths(answer: str) -> tuple[bool, str]:
    near = " ".join(
        answer[max(0, m.start() - 300) : m.end() + 300]
        for m in re.finditer(r"width|k_max", answer, re.IGNORECASE)
    )
    found = {w for w in ("4", "6") if re.search(rf"\b{w}\b", near)}
    return found == {"4", "6"}, f"widths near 'width': {sorted(found) or 'none'}"


def check(answer: str, transcript: str, work: Path) -> dict:
    return {
        "saves the converted circuit": _converted_circuit(work),
        "samples the stand-in with Stim": _checked_against_stim(answer, transcript),
        "checks the noiseless real-T circuit": (
            "without_noise" in transcript,
            (
                "without_noise used"
                if "without_noise" in transcript
                else "no noiseless check"
            ),
        ),
        "reports widths 4 and 6": _reports_widths(answer),
        "notes the stand-in observable is random": answer_matches(
            answer,
            r"random|non-?deterministic|not deterministic|coin",
            "the random observable",
        ),
        "postselects inside Tsim": postselects_in_tsim(transcript),
        "no stabilizer-rank strategy": no_stabilizer_rank(transcript),
    }

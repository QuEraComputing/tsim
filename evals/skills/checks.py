"""Deterministic checks shared by the eval cases.

Every check takes the agent's final answer, its transcript (all commands and code it ran or
wrote, as text) and its working directory, and returns ``(passed, evidence)``.
"""

from __future__ import annotations

import re
from pathlib import Path

Result = tuple[bool, str]


def _find(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    m = re.search(pattern, text, flags)
    return (
        None
        if m is None
        else text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
    )


def no_stabilizer_rank(transcript: str) -> Result:
    hit = _find(r"strategy\s*=\s*['\"](cat5|bss|cutting)['\"]", transcript)
    return hit is None, hit or "no stabilizer-rank strategy used"


def uses_detector_sampler(transcript: str) -> Result:
    hit = _find(r"compile_detector_sampler\(", transcript)
    return hit is not None, hit or "compile_detector_sampler never called"


def postselects_in_tsim(transcript: str) -> Result:
    hit = _find(r"postselection_mask\s*=", transcript)
    return hit is not None, hit or "postselection_mask never passed to sample()"


def no_sinter_collect(transcript: str) -> Result:
    hit = _find(r"sinter\.collect\(", transcript)
    return hit is None, hit or "sinter.collect not used"


def answer_matches(answer: str, pattern: str, what: str) -> Result:
    hit = _find(pattern, answer)
    return hit is not None, hit or f"answer does not mention {what}"


def writes_json_with(work: Path, key: str) -> Result:
    for path in work.rglob("*.json"):
        if key in path.read_text(errors="ignore"):
            return True, str(path.relative_to(work))
    return False, f"no JSON file containing {key!r}"

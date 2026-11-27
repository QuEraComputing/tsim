#!/usr/bin/env python3
"""PreToolUse hook for Bash: block two commands that have taken sessions down.

Exit code 2 blocks the command and shows stderr to the agent. Everything else
passes. Keep this list short: it is for rules that must hold every time.
"""

import json
import re
import sys

COMPUTE = re.compile(r"\b(python3?|pytest|uv run)\b")
CAPPED = re.compile(r"\b(tsim-dev|systemd-run)\b")
DETACHED = re.compile(r"\bnohup\b|\bsetsid\b|&\s*(disown\b|$)", re.MULTILINE)


def verdict(tool_input: dict) -> str | None:
    """Return why the command is blocked, or None to let it run."""
    command = tool_input.get("command", "")
    if re.search(r"\bpkill\b[^|;&\n]*\s(-\w*f|--full\b)", command):
        return (
            "`pkill -f` also matches this shell's own command line and kills it. "
            "Stop a job with `uv run tsim-dev job stop NAME`, or kill by PID."
        )
    background = tool_input.get("run_in_background") or DETACHED.search(command)
    if background and COMPUTE.search(command) and not CAPPED.search(command):
        return (
            "Background compute needs a hard memory cap; uncapped jobs have used "
            "55 GB and killed every session on this machine. Start it with "
            "`uv run tsim-dev job start NAME -- <command>` and follow it with "
            "`uv run tsim-dev job wait NAME`."
        )
    return None


def main() -> int:
    """Read the hook payload from stdin and decide."""
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    reason = verdict(payload.get("tool_input") or {})
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

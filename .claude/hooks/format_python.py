#!/usr/bin/env python3
"""PostToolUse hook for Edit and Write: format an edited Python file at once.

pre-commit reformats files only at commit time, which left agents editing text
that black had since rewritten. Running the same pinned isort and black right
after each edit keeps the file in its final form, and Claude Code shows the agent
the result. Never fails the tool call: formatting errors surface at commit time.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HOOKS = ("isort", "black")


def main() -> int:
    """Read the hook payload from stdin and format the edited file."""
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")).resolve()
    path = Path(payload.get("tool_input", {}).get("file_path", "")).resolve()
    if path.suffix != ".py" or not path.is_file() or not path.is_relative_to(root):
        return 0
    for hook in HOOKS:
        # NOTE: pre-commit exits 1 when it changes the file; that is not an error.
        subprocess.run(
            [
                "uv",
                "run",
                "pre-commit",
                "run",
                hook,
                "--files",
                str(path.relative_to(root)),
            ],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

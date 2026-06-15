# Contributing to Tsim

Thanks for your interest in contributing. This document explains how
contributions are licensed and how to get a change merged.

## Licensing of contributions

By opening a pull request against this repository, **you agree that your
contribution is licensed under the [Apache License, Version 2.0](LICENSE)
and that you accept the terms of the [Tsim Contributor License
Agreement](CLA.md)**, which grants QuEra Computing Inc. and downstream
recipients the rights described in that document.

You retain copyright on your contribution; the CLA grants licenses, it
does not assign ownership. See [`CLA.md`](CLA.md) for the full text. If
you are contributing on behalf of an employer, please make sure your
employer is aware of and authorizes the contribution — the CLA's
representations cover this in §4.

If you do not agree to those terms, please do not open a pull request.

## Workflow

1. Fork the repository and create a topic branch off `main`.
2. Install dependencies with `uv sync`, then set up the hooks with
   `pre-commit install` (first time only).
3. Make your change. Keep commits focused and use
   [Conventional Commits](https://www.conventionalcommits.org/)
   (e.g. `feat: add new gate`).
4. Add or update tests for the behavior you changed.
5. Run the hooks and the test suite locally before pushing:

   ```bash
   pre-commit run --all-files   # black, isort, ruff, pyright, codespell, nbstripout
   uv run pytest                # run the test suite
   ```

6. Open a pull request. CI runs the pre-commit checks and the test suite.
   First-time contributors will see an automated welcome comment with
   a pointer to this document.
7. A maintainer will review. Be prepared to iterate.

## Reporting issues

Use GitHub Issues for bugs and feature requests. For security-sensitive
reports, please email the maintainers privately rather than filing a
public issue.

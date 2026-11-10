coverage-run:
    coverage run -m pytest test

coverage-xml:
    coverage xml

coverage-html:
    coverage html

coverage-report:
    coverage report

coverage-open:
    open htmlcov/index.html

coverage-unit:
    coverage run -m pytest test/unit
    coverage report --include="**/types.py","**/taskgen.py","**/concrete.py"

coverage: coverage-run coverage-xml coverage-report

# Build the docs and serve them locally (needs `npm i -g mint`).
doc:
    uv run python docs/build.py
    cd docs && mint dev

# Render notebooks from their stored outputs instead of re-executing them.
doc-quick:
    uv run python docs/build.py --no-execute
    cd docs && mint dev

# Build the docs and run the same checks as CI.
doc-build:
    uv run python docs/build.py
    cd docs && mint validate && mint broken-links --check-anchors --check-redirects --check-snippets

vulture:
    uv run vulture

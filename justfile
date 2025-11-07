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

doc:
    mkdocs serve

doc-build:
    mkdocs build

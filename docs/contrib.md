# Contributing

Please see [Installation](install.md) for instructions on how to set up your development environment.

## Pre-commit hooks

We use `pre-commit` to run formatting, linter and type checks before you commit your changes. The pre-commit hooks are installed as part of the development dependencies. You can setup `pre-commit` using the following command:

```bash
pre-commit install
```

If the checks fail, the commit will be rejected.

## Running the tests

We use `pytest` for testing. To run the tests, simply run:

```bash
uv run pytest
```


## Code style

We use `black` for code formatting. Besides the linter requirements, we also require the following
good-to-have practices:

### Naming

- try not to use abbreviation as names, unless it's a common abbreviation like `idx` for `index`
- try not create a lot of duplicated name prefix unless the extra information is necessary when accessing the class object.
- use `snake_case` for naming variables and functions, and `CamelCase` for classes.

### Comments

- try not to write comments, unless it's really necessary. The code should be self-explanatory.
- if you have to write comments, try to use `NOTE:`, `TODO:` `FIXME:` tags to make it easier to search for them.

## Documentation

We use `just` for managing command line tools and scripts. It should be installed when you run `uv sync`. To build the documentation, simply run:

```bash
uv run just doc
```

This will launch a local server to preview the documentation. You can also run `uv run just doc-build` to build the documentation without launching the server.

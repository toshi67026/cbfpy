# cbfpy
Python package for using simple control barrier function.

[**Documentation**](https://toshi67026.github.io/cbfpy/)

## Requirements
- Python >= 3.10

## Installation
```sh
pip install -e .
```

With dev/test dependencies:
```sh
pip install -e ".[dev,test]"
```

## Examples
<img src=assets/example_circle_cbf.gif width=50%><img src=assets/example_pnorm2d_cbf.gif width=50%>
<img src=assets/example_unicycle_circle_cbf.gif width=50%><img src=assets/example_unicycle_pnorm2d_cbf.gif width=50%>
<img src=assets/example_scalar_cbf.gif width=50%><img src=assets/example_scalar_range_cbf.gif width=50%>
<img src=assets/example_lidar_cbf.gif width=50%>

### Usage
```sh
python examples/example_{cbf name}.py
```

## Documentation
Documentation is automatically published to [GitHub Pages](https://toshi67026.github.io/cbfpy/) on push to main.

To build locally:
```sh
pip install -e ".[dev]"
sphinx-build docs/source docs/build
```
Then open `docs/build/index.html`.

## Tools
### Format & Lint
- [ruff](https://docs.astral.sh/ruff/) (format + lint + import sort)
- [mypy](https://mypy-lang.org/) (type check)
```sh
ruff format .
ruff check --fix .
mypy .
```

### Test
```sh
pytest
```
The vscode extension [coverage-gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) can be used to display the test coverage.

### pre-commit
Apply [config file](.pre-commit-config.yaml) for pre-commit.
```sh
pre-commit install
```

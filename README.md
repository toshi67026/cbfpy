# cbfpy
Python package for using simple control barrier function.

## Requirements
- poetry 1.3.1

## Installation
Create virtualenv and install dependencies defined for the project.
```sh
poetry install
```

Build and install locally with pip.
```sh
poetry build
python -m pip install cbfpy --find-links=dist
```

## Examples
<img src=assets/example_circle_cbf.gif width=50%><img src=assets/example_pnorm2d_cbf.gif width=50%>
<img src=assets/example_unicycle_circle_cbf.gif width=50%><img src=assets/example_unicycle_pnorm2d_cbf.gif width=50%>
<img src=assets/example_scalar_cbf.gif width=50%><img src=assets/example_scalar_range_cbf.gif width=50%>
<img src=assets/example_lidar_cbf.gif width=50%>

### Usage
```sh
poetry run python examples/example_{cbf name}.py
```

## Document
Generate document from docstring.
```sh
poetry run task docs
```

Browse generated document by opening the html files in docs/build/ from your browser.
<img src=assets/sphinx_cbfpy.png>

## Tools
### Format
- isort
- black
```sh
poetry run task fmt
```

### Lint
- black
- ruff
- mypy
```sh
poetry run task lint
```

### Test
- pytest
- pytest-cov
```sh
poetry run task test
```
The vscode extension [coverage-gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters) can be used to display the test coverage.

### pre-commit
Apply [config file](.pre-commit-config.yaml) for pre-commit.
```sh
poetry run pre-commit install
```

### Export requirements.txt
```sh
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

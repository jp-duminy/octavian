# Installation

Octavius is available on [pip](https://pypi.org/project/octavius/). 

```bash
pip install octavius
```

For photometry, please specify the optional dependency as such:

```bash
pip install "octavius[photometry]"
```

For developers, please either clone or fork and specify the developer tools.

```bash
pip install -e ".[dev,docs,test]"
```

The package is currently pinned to relatively recent versions of its dependencies. [uv](https://docs.astral.sh/uv/) provides a fast, user-friendly installation route and methods for integration into existing projects.

```bash
uv pip install octavius
uv add octavius
```

Packaging on `conda-forge` is planned in the near future.
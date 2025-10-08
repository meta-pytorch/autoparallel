# AutoParallel

WARNING: Highly under development!

This currently works on PyTorch 2.8.0.dev20250506.

## Installing it

```
pip install git+ssh://git@github.com/pytorch-labs/autoparallel.git
```

## Developing it
```
cd autoparallel
pip install -e .
```
Modified Python files will be reflected immediately.

Run linter before submitting the PR
```
pip install pre-commit
pre-commit run --all-files
```

If you got ``An unexpected error has occurred: ... 'python3.11')``, try modify `.pre-commit-config.yaml`/`language_version: python3.11` to match your python version.

## Running it

```
python examples/example_autoparallel.py
```

## License

Autoparallel is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.

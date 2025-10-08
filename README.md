# AutoParallel

> ⚠️ **Early Development Warning** Autoparallel is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

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

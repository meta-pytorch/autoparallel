# AutoParallel

AutoParallel is a PyTorch library that automatically shards and parallelizes
models for distributed training. Given a model and a device mesh, it uses linear
programming to find an optimal sharding strategy (FSDP, tensor parallelism, or a
mix) and applies it — no manual parallelism code required.

> **Early Development Warning** — AutoParallel is experimental. Expect bugs,
> incomplete features, and APIs that may change. Bugfixes are welcome; please
> discuss significant changes in the
> [issue tracker](https://github.com/pytorch-labs/autoparallel/issues) before
> starting work.

## Requirements

- Python >= 3.10
- [PyTorch nightly](https://pytorch.org/get-started/locally/) (newer than 2.10)

## Installing

```bash
# Via SSH
pip install git+ssh://git@github.com/pytorch-labs/autoparallel.git

# Via HTTPS
pip install git+https://github.com/pytorch-labs/autoparallel.git
```

## Quick start

The simplest way to try AutoParallel is with a HuggingFace model. This runs
entirely on a single machine using a fake process group — no multi-GPU setup
needed:

```bash
pip install transformers
python examples/example_hf.py --model gpt2 --mesh 8
```

You should see log output ending with `Forward + backward OK`.

For more examples (LLaMA-3, pipeline parallelism, distributed checkpointing),
see the [`examples/`](examples/) directory.

## Developing

```bash
cd autoparallel
pip install -e .
```

Modified Python files are reflected immediately.

Run linters before submitting a PR:

```bash
pip install pre-commit
pre-commit run --all-files
```

Run tests (requires a CUDA GPU):

```bash
pip install -r requirements-test.txt
pytest tests/
```

## License

AutoParallel is BSD-3 licensed, as found in the [LICENSE](LICENSE) file.

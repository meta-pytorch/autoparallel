# Measurement lifecycle

`python -m harness.cli run --model MODEL --setting SETTING` is the supported
entry point. It resolves the locked sources, runtime, assets, campaign, and
matrix point before invoking this lifecycle.

`measurement.py` remains an internal stage driver for package/dry-run, submit,
monitor, stable OilFS retrieval, and analysis. Its individual subcommands are
retained for diagnosis and recovery; they are not an alternate way to select
source or runtime versions.

The submit stage sets CRITICAL/99 and verifies the scheduler readback. Retrieval
is permitted only after the root job, latest attempt, task groups, and tasks all
report COMPLETE with no continuity errors.

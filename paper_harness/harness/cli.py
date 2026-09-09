from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .analysis import analyze_campaign
from .campaign import CampaignError, load_campaign
from .dryrun import audit_dryrun
from .packaging import package_campaign
from .sources import manifest_digest, tree_manifest
from .validation import validate_campaign


MAST_HANDLE_PATTERN = re.compile(r"mast_conda://[^/\s`]+/([^\s`]+)")


def _asset_roots(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise CampaignError("--asset-root requires NAME=/absolute/path")
        name, path = value.split("=", 1)
        if not name or not Path(path).is_absolute():
            raise CampaignError("--asset-root requires NAME=/absolute/path")
        result[name] = Path(path)
    return result


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--autoparallel-root", type=Path, required=True)
    parser.add_argument("--asset-root", action="append", default=[])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))


def _torchx_command(attempt: Path, *, dryrun: bool) -> list[str]:
    resolved = json.loads((attempt / "validation/resolved_campaign.json").read_text())
    mast = resolved["mast"]
    payload = attempt / "package/payload"
    command = ["torchx", "run"]
    if dryrun:
        command.append("--dryrun")
    command.extend(
        [
            f"--scheduler_args=conda_fbpkg_id={mast['conda_fbpkg']},localityConstraints={mast['locality']},forceSingleRegion=False",
            "mast.py:train",
            "--name",
            resolved["name"],
            "--h",
            mast["hardware"],
            "--nodes",
            str(mast["nodes"]),
            "--nproc_per_node",
            str(mast["nproc_per_node"]),
            "--retries",
            str(mast.get("retries", 0)),
            "--payload_root",
            str(payload),
        ]
    )
    return command


def _verify_sealed_attempt(attempt: Path) -> dict:
    package_report = json.loads((attempt / "package_report.json").read_text())
    expected = package_report["payload_tree_sha256"]
    sealed = (attempt / "SEALED").read_text().strip()
    actual = manifest_digest(tree_manifest(attempt / "package/payload"))
    if expected != sealed or expected != actual:
        raise CampaignError(
            f"attempt payload changed after packaging: expected={expected}, actual={actual}"
        )
    return package_report


def _submitted_job_id(output: str) -> str:
    matches = sorted(set(MAST_HANDLE_PATTERN.findall(output)))
    if len(matches) != 1:
        raise CampaignError(
            f"expected one submitted MAST job handle, found {len(matches)}"
        )
    return matches[0]


def _run_torchx(attempt: Path, *, dryrun: bool) -> dict:
    package_report = _verify_sealed_attempt(attempt)
    if not package_report["validation"].get("probe_configs"):
        raise CampaignError(
            "MAST rendering/submission requires ConfigManager validation in the "
            "exact compatible Python environment"
        )
    if not dryrun:
        dryrun_path = attempt / "dryrun_audit.json"
        if not dryrun_path.is_file():
            raise CampaignError("submission requires a passing dry-run audit")
        dryrun_audit = json.loads(dryrun_path.read_text())
        if dryrun_audit.get("status") != "passed":
            raise CampaignError("submission requires a passing dry-run audit")
        launcher_digest = manifest_digest(
            tree_manifest(Path(__file__).resolve().parents[1] / "launcher")
        )
        if launcher_digest != dryrun_audit.get("workspace_tree_sha256"):
            raise CampaignError(
                "launcher changed after the audited dry run; create a new attempt"
            )
        if dryrun_audit.get("payload_tree_sha256") != package_report[
            "payload_tree_sha256"
        ]:
            raise CampaignError(
                "dry-run payload does not match the sealed submission payload"
            )
    command = _torchx_command(attempt, dryrun=dryrun)
    prefix = "dryrun" if dryrun else "submission"
    (attempt / f"{prefix}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1] / "launcher",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (attempt / f"{prefix}.stdout").write_text(completed.stdout)
    (attempt / f"{prefix}.stderr").write_text(completed.stderr)
    (attempt / f"{prefix}.returncode").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise CampaignError(f"{' '.join(command)} exited {completed.returncode}")
    if dryrun:
        return audit_dryrun(
            attempt,
            stdout=completed.stdout,
            stderr=completed.stderr,
            launcher_root=Path(__file__).resolve().parents[1] / "launcher",
        )
    job_id = _submitted_job_id(completed.stdout + "\n" + completed.stderr)
    handle = f"mast_conda://torchx/{job_id}"
    (attempt / "job_id.txt").write_text(f"{job_id}\n")
    result = {
        "status": "submitted",
        "command": command,
        "job_id": job_id,
        "handle": handle,
        "stdout": completed.stdout,
    }
    (attempt / "submission.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("campaign", type=Path)
    validate.add_argument("--point")
    validate.add_argument("--mode", choices=("formal", "gate"), default="formal")
    validate.add_argument("--output", type=Path, required=True)
    _add_sources(validate)

    package = subparsers.add_parser("package")
    package.add_argument("campaign", type=Path)
    package.add_argument("--point")
    package.add_argument("--mode", choices=("formal", "gate"), default="formal")
    package.add_argument("--attempt", type=Path, required=True)
    _add_sources(package)

    for name in ("render-mast", "submit"):
        command = subparsers.add_parser(name)
        command.add_argument("--attempt", type=Path, required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("campaign", type=Path)
    analyze.add_argument("--point")
    analyze.add_argument("--mode", choices=("formal", "gate"), default="formal")
    analyze.add_argument("--attempt-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--tlparse-bin", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            campaign = load_campaign(args.campaign, point=args.point, mode=args.mode)
            assets = _asset_roots(args.asset_root)
            result = validate_campaign(
                campaign,
                torchtitan_root=args.torchtitan_root,
                autoparallel_root=args.autoparallel_root,
                output_dir=args.output,
                asset_roots=assets,
                python=args.python,
            )
        elif args.command == "package":
            campaign = load_campaign(args.campaign, point=args.point, mode=args.mode)
            result = package_campaign(
                campaign,
                torchtitan_root=args.torchtitan_root,
                autoparallel_root=args.autoparallel_root,
                attempt_root=args.attempt,
                asset_roots=_asset_roots(args.asset_root),
                python=args.python,
            )
        elif args.command in {"render-mast", "submit"}:
            result = _run_torchx(
                args.attempt.resolve(), dryrun=args.command == "render-mast"
            )
        else:
            campaign = load_campaign(args.campaign, point=args.point, mode=args.mode)
            result = analyze_campaign(
                campaign,
                attempt_root=args.attempt_root,
                output_dir=args.output,
                tlparse_bin=args.tlparse_bin,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

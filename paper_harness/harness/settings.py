from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .campaign import CampaignError

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "run_settings.toml"


@dataclass(frozen=True)
class RunSetting:
    model: str
    setting: str
    campaign: Path
    point: str | None


def load_run_settings(path: Path = SETTINGS_PATH) -> dict[tuple[str, str], RunSetting]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw.get("schema_version") != 1:
        raise CampaignError("run_settings.toml schema_version must be 1")
    root = path.resolve().parent
    result: dict[tuple[str, str], RunSetting] = {}
    for row in raw.get("settings", []):
        if not isinstance(row, dict):
            raise CampaignError("each run setting must be a table")
        model = str(row.get("model", ""))
        setting = str(row.get("setting", ""))
        campaign = root / str(row.get("campaign", ""))
        key = (model, setting)
        if not model or not setting or key in result:
            raise CampaignError(f"invalid or duplicate run setting {key!r}")
        if not campaign.is_file() or not campaign.is_relative_to(root):
            raise CampaignError(f"run setting campaign does not exist: {campaign}")
        point = row.get("point")
        if point is not None and not isinstance(point, str):
            raise CampaignError(f"run setting point must be a string: {key!r}")
        result[key] = RunSetting(model, setting, campaign, point)
    if not result:
        raise CampaignError("run_settings.toml contains no settings")
    return result


def resolve_run_setting(model: str, setting: str) -> RunSetting:
    settings = load_run_settings()
    try:
        return settings[(model, setting)]
    except KeyError as error:
        available = sorted(
            value.setting for value in settings.values() if value.model == model
        )
        raise CampaignError(
            f"unknown model/setting {(model, setting)!r}; available settings: {available}"
        ) from error

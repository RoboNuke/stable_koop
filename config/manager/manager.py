"""YAML-driven config manager (copied from failure_prevention_curriculum).

Loads a single YAML file that contains one section per registered config dataclass
(top-level header == header name in :attr:`ConfigManager.REGISTRY`). Each section's
keys must match dataclass fields exactly; unknown keys raise. Nested dataclass-typed
fields recurse into their own dict.

Strict by design — any divergence between the YAML and the registered dataclasses
is a hard error, never a silent fallback.

The stable_koop pipeline runs each workflow stage from its own YAML (one header
per file), so :meth:`ConfigManager.load_stage` is provided as a thin
per-header convenience over the strict whole-YAML :meth:`load`.
"""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any

import yaml

from config.manager.gather_data_cfg import GatherDataCfg
from config.manager.train_koopman_cfg import TrainKoopmanCfg
from config.manager.lqr_controller_cfg import LQRControllerCfg
from config.manager.train_residual_cfg import TrainResidualCfg
from config.manager.eval_cfg import EvalCfg


def _resolved_field_types(dataclass_type: type) -> dict[str, Any]:
    """Resolve a dataclass's annotations against the defining module's globals.

    ``from __future__ import annotations`` makes ``field.type`` a string;
    ``typing.get_type_hints`` evaluates those strings in the module they were
    defined in, which has the right imports in scope.
    """
    return typing.get_type_hints(dataclass_type)


def _to_yaml_safe(obj: Any) -> Any:
    """Recursively convert a Python value to YAML-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_yaml_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(x) for x in obj]
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if callable(obj):
        mod = getattr(obj, "__module__", "<unknown>")
        name = getattr(obj, "__qualname__", repr(obj))
        return f"{mod}.{name}"
    return repr(obj)


class ConfigManager:
    """Header → dataclass dispatch table for stable_koop's five workflow stages."""

    REGISTRY: dict[str, type] = {
        "gather_data_cfg": GatherDataCfg,
        "train_koopman_cfg": TrainKoopmanCfg,
        "lqr_controller_cfg": LQRControllerCfg,
        "train_residual_cfg": TrainResidualCfg,
        "eval_cfg": EvalCfg,
    }

    @classmethod
    def load(cls, yaml_path: str | Path | None) -> dict:
        """Return ``{header: populated_instance}`` for every registered header.

        ``yaml_path is None`` returns one default-constructed instance per header.
        Otherwise, every registered header MUST appear; unknown headers, unknown
        fields, and missing files all raise.
        """
        if yaml_path is None:
            return {header: type_() for header, type_ in cls.REGISTRY.items()}

        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root in {path} must be a mapping, got {type(raw).__name__}"
            )

        unknown_headers = set(raw) - set(cls.REGISTRY)
        if unknown_headers:
            raise KeyError(
                f"Unknown config header(s) {sorted(unknown_headers)} in {path}; "
                f"expected one of {sorted(cls.REGISTRY)}"
            )
        missing_headers = set(cls.REGISTRY) - set(raw)
        if missing_headers:
            raise KeyError(
                f"Missing config section(s) {sorted(missing_headers)} in {path}"
            )

        out = {}
        for header, type_ in cls.REGISTRY.items():
            out[header] = cls._build(type_, raw[header], context=header)
        return out

    @classmethod
    def load_stage(cls, yaml_path: str | Path, header: str):
        """Load a single stage's dataclass instance from a YAML file.

        The YAML may carry the requested ``header`` either alone or alongside
        other known headers (the latter is how a combined train_koopman + LQR
        YAML works). Unknown headers raise. Strict on field names just like
        :meth:`load`.
        """
        if header not in cls.REGISTRY:
            raise KeyError(
                f"Unknown stage header {header!r}; expected one of {sorted(cls.REGISTRY)}"
            )
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root in {path} must be a mapping, got {type(raw).__name__}"
            )
        if header not in raw:
            raise KeyError(
                f"Header {header!r} not found in {path}; got {sorted(raw)}"
            )
        unknown = set(raw) - set(cls.REGISTRY)
        if unknown:
            raise KeyError(
                f"Unknown header(s) {sorted(unknown)} in {path}; "
                f"expected only registered headers {sorted(cls.REGISTRY)}"
            )
        return cls._build(cls.REGISTRY[header], raw[header], context=header)

    @classmethod
    def dump(cls, configs: dict, path: str | Path) -> None:
        """Write a loaded-configs dict to a YAML file at ``path``."""
        if not isinstance(configs, dict):
            raise TypeError(f"configs must be a dict, got {type(configs).__name__}")
        missing = set(cls.REGISTRY) - set(configs)
        if missing:
            raise KeyError(
                f"Configs missing required headers: {sorted(missing)}; "
                f"expected all of {sorted(cls.REGISTRY)}"
            )
        extra = set(configs) - set(cls.REGISTRY)
        if extra:
            raise KeyError(
                f"Configs has unexpected headers: {sorted(extra)}; "
                f"expected only {sorted(cls.REGISTRY)}"
            )

        serializable = {header: _to_yaml_safe(configs[header]) for header in cls.REGISTRY}
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            yaml.safe_dump(serializable, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def dump_stage(cls, instance, header: str, path: str | Path) -> None:
        """Write a single stage's dataclass instance to a per-stage YAML."""
        if header not in cls.REGISTRY:
            raise KeyError(
                f"Unknown stage header {header!r}; expected one of {sorted(cls.REGISTRY)}"
            )
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            yaml.safe_dump(
                {header: _to_yaml_safe(instance)},
                f,
                sort_keys=False,
                default_flow_style=False,
            )

    @classmethod
    def _build(cls, dataclass_type: type, data: Any, *, context: str):
        """Recursively populate ``dataclass_type`` from a YAML-derived dict.

        Strict: every key in ``data`` must be a declared field; any extras raise.
        Nested dataclass-typed fields recurse into their own dicts.
        """
        if not dataclasses.is_dataclass(dataclass_type):
            raise TypeError(
                f"{context}: registered type {dataclass_type!r} is not a dataclass"
            )
        if not isinstance(data, dict):
            raise ValueError(
                f"{context}: expected mapping, got {type(data).__name__}"
            )

        valid_names = {f.name for f in dataclasses.fields(dataclass_type)}
        unknown = set(data) - valid_names
        if unknown:
            raise KeyError(
                f"Unknown field(s) {sorted(unknown)} under '{context}'; "
                f"expected one of {sorted(valid_names)}"
            )

        type_hints = _resolved_field_types(dataclass_type)
        kwargs = {}
        for name, value in data.items():
            field_type = type_hints.get(name)
            if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
                kwargs[name] = cls._build(field_type, value, context=f"{context}.{name}")
            else:
                kwargs[name] = value
        return dataclass_type(**kwargs)

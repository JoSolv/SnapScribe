"""Generic dataclass <-> JSON-compatible dict conversion.

Keeps the core data model free of pydantic (profiles use pydantic, where
validation of user-authored YAML is the point) while still giving the session
a lossless JSON round-trip.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import types
import typing
from pathlib import Path, PurePath
from typing import Any, TypeVar, Union, get_args, get_origin

T = TypeVar("T")

_NONE_TYPE = type(None)


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums/paths/datetimes into JSON types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(v) for v in value]
    return value


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not _NONE_TYPE]
        optional = len(args) != len(get_args(tp))
        if len(args) == 1:
            return args[0], optional
        return tp, optional
    return tp, False


def from_jsonable(tp: Any, value: Any) -> Any:
    """Rebuild a value of (possibly generic) type ``tp`` from JSON data."""
    tp, optional = _unwrap_optional(tp)
    if value is None:
        return None if optional else _empty_for(tp)

    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return from_dict(tp, value or {})

    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(value)
        except ValueError:
            return next(iter(tp))

    if tp in (dt.datetime,):
        return dt.datetime.fromisoformat(value) if isinstance(value, str) else value
    if tp in (dt.date,):
        return dt.date.fromisoformat(value) if isinstance(value, str) else value
    if isinstance(tp, type) and issubclass(tp, PurePath):
        return Path(value)

    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (list, set, frozenset):
        inner = args[0] if args else Any
        items = [from_jsonable(inner, v) for v in (value or [])]
        return origin(items) if origin is not list else items
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_jsonable(args[0], v) for v in (value or []))
        return tuple(from_jsonable(a, v) for a, v in zip(args, value or [], strict=False))
    if origin is dict:
        kt = args[0] if args else str
        vt = args[1] if len(args) > 1 else Any
        out = {}
        for k, v in (value or {}).items():
            key = int(k) if kt is int else k
            out[key] = from_jsonable(vt, v)
        return out

    return value


def _empty_for(tp: Any) -> Any:
    origin = get_origin(tp) or tp
    if origin in (list, set, tuple):
        return origin()
    if origin is dict:
        return {}
    if tp is str:
        return ""
    if tp is int:
        return 0
    if tp is float:
        return 0.0
    if tp is bool:
        return False
    return None


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate dataclass ``cls`` from ``data``, ignoring unknown keys."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name not in data:
            continue
        kwargs[f.name] = from_jsonable(hints.get(f.name, Any), data[f.name])
    return cls(**kwargs)  # type: ignore[return-value]

"""Pure-Python parser for Snowdrop textual ``.mshader`` declarations."""

from __future__ import annotations

import re


_SAMPLER_RE = re.compile(
    r"MR_Sampler2D\s+(?P<field>\w+)\s*:\s*MR_Texture(?P<texidx>\d+)\s*"
    r"(?:\{(?P<block>[^{}]*)\}\s*)?"
    r"(?:<(?P<meta>[^<>]*)>\s*)?;",
    re.DOTALL,
)
_TEXTURE_RE = re.compile(r'texture\s*=\s*"([^"]*)"')
_PIN_ID_RE = re.compile(r"pinId\s*=\s*(\d+)")
_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
_SHADER_TYPE_RE = re.compile(r'shaderType\s*=\s*"([^"]*)"')
_PARAMETER_RE = re.compile(
    r"(?P<type>(?:float|half|int|uint|bool)[1-4]?)\s+"
    r"(?P<field>\w+)\s*<(?P<meta>[^<>]*)>\s*;",
    re.DOTALL,
)
_DEFAULT_RE = re.compile(r"default\s*=\s*([^,>]+)")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _normalise_path(value):
    return str(value or "").replace("\\", "/").lstrip("/")


def _sampler_role(field):
    key = field.casefold()
    if key in {"color", "decal", "diffuse"}:
        return "d"
    if key == "normal":
        return "n"
    if key == "material":
        return "m"
    if "detailnormal" in key or key == "detailsampler":
        return "detail_normal"
    return key


def parse_shader_source(data):
    """Parse sampler and numeric-parameter metadata from an ``.mshader``."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    shader_type = _SHADER_TYPE_RE.search(text)
    samplers = []
    for match in _SAMPLER_RE.finditer(text):
        block = match.group("block") or ""
        metadata = match.group("meta") or ""
        texture = _TEXTURE_RE.search(block)
        pin_id = _PIN_ID_RE.search(metadata)
        label = _LABEL_RE.search(metadata)
        samplers.append({
            "field": match.group("field"),
            "texture_index": int(match.group("texidx")),
            "role": _sampler_role(match.group("field")),
            "default_texture": _normalise_path(texture.group(1)) if texture else "",
            "pin_id": int(pin_id.group(1)) if pin_id else None,
            "label": label.group(1) if label else "",
            "graph_connectable": pin_id is not None,
        })
    parameters = []
    for match in _PARAMETER_RE.finditer(text):
        metadata = match.group("meta") or ""
        pin_id = _PIN_ID_RE.search(metadata)
        if pin_id is None:
            continue
        label = _LABEL_RE.search(metadata)
        default_match = _DEFAULT_RE.search(metadata)
        default = None
        if default_match:
            raw_default = default_match.group(1).strip()
            lowered = raw_default.casefold()
            if lowered in {"true", "false"}:
                default = lowered == "true"
            else:
                values = [float(value) for value in _NUMBER_RE.findall(raw_default)]
                if values:
                    default = values[0] if len(values) == 1 else values
        parameters.append({
            "field": match.group("field"),
            "value_type": match.group("type"),
            "pin_id": int(pin_id.group(1)),
            "label": label.group(1) if label else "",
            "default": default,
        })
    return {
        "shader_type": shader_type.group(1) if shader_type else "",
        "samplers": samplers,
        "parameters": parameters,
    }


__all__ = ("parse_shader_source",)

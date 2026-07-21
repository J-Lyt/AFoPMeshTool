"""Read Snowdrop banshee colour-pattern manifests and their members.

The text layout and field meanings were cross-checked against PandoraPaint's
``patterns.py``.  The add-on only needs the read path, so this module deliberately
does not mint UIDs or write game assets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_COLOR_HEADER = re.compile(
    r'^\s*ColorPattern\s+"(?P<name>[^"]*)"\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$'
)
_COLOR_VALUE = re.compile(
    r"^\s*myColor(?P<index>\d+)\s+0x(?P<value>[0-9A-Fa-f]{8})\s*$"
)
_CONTROL_HEADER = re.compile(
    r'^\s*PatternControl\s+"(?P<name>[^"]*)"\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$'
)
_CONTROL_VALUE = re.compile(
    r"^\s*(?P<key>my\w+)\s+(?P<value>-?\d+(?:\.\d+)?)\s*$"
)
_DATA_INCLUDE = re.compile(r'^\s*include\s+(?:"(?P<quoted>[^"]+)"|(?P<plain>.+?))\s*$')
_DATA_HEADER = re.compile(
    r'^\s*BansheePatternData\s+(?:"(?P<quoted>[^"]+)"|(?P<plain>\S+))'
    r"\s*<\s*uid=(?P<uid>[0-9A-Fa-f]+)\s*>\s*$"
)
_DATA_REF = re.compile(
    r'^\s*(?P<key>my\w+)\s*<\s*uid=[0-9A-Fa-f]+\s*>\s*=\s*'
    r'"(?P<name>[^"]*)"\s+(?P<target>[0-9A-Fa-f]+)\s*$'
)
_DATA_COAT = re.compile(r"^\s*(?P<key>my\w+Coat)\s+(?P<path>\S+)\s*$")

_CONTROL_FIELDS = {
    "myPattern1Invert": "invert1",
    "myPattern2Invert": "invert2",
    "myPattern1LevelControl": "level1",
    "myPattern2LevelControl": "level2",
}
_REFERENCE_FIELDS = {
    "myBodyColorPattern": ("body", "color"),
    "myHeadColorPattern": ("head", "color"),
    "myBodyPatternControl": ("body", "control"),
    "myHeadPatternControl": ("head", "control"),
}


def _text(data):
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)


@dataclass(frozen=True)
class ColorPattern:
    name: str
    uid: str
    colors: tuple[int, ...]

    @classmethod
    def loads(cls, data):
        name = uid = None
        colors = [0xFF000000] * 10
        for line in _text(data).splitlines():
            header = _COLOR_HEADER.match(line)
            if header:
                name, uid = header["name"], header["uid"]
                continue
            value = _COLOR_VALUE.match(line)
            if value:
                index = int(value["index"])
                if 1 <= index <= 10:
                    colors[index - 1] = int(value["value"], 16)
        if name is None:
            raise ValueError("not a valid .mcolorpattern")
        return cls(name, uid or "", tuple(colors))

    def rgb(self, index):
        value = self.colors[index]
        return (
            ((value >> 16) & 0xFF) / 255.0,
            ((value >> 8) & 0xFF) / 255.0,
            (value & 0xFF) / 255.0,
        )


@dataclass(frozen=True)
class PatternControl:
    name: str
    uid: str
    invert1: float = 1.0
    invert2: float = 1.0
    level1: float = 1.0
    level2: float = 1.0

    @classmethod
    def loads(cls, data):
        name = uid = None
        values = {}
        for line in _text(data).splitlines():
            header = _CONTROL_HEADER.match(line)
            if header:
                name, uid = header["name"], header["uid"]
                continue
            value = _CONTROL_VALUE.match(line)
            if value and value["key"] in _CONTROL_FIELDS:
                values[_CONTROL_FIELDS[value["key"]]] = float(value["value"])
        if name is None:
            raise ValueError("not a valid .mpatterncontrol")
        return cls(name=name, uid=uid or "", **values)


@dataclass(frozen=True)
class BansheePatternData:
    name: str
    uid: str
    includes: tuple[str, ...]
    references: tuple[tuple[str, str, str, str], ...]
    body_coat: str | None
    head_coat: str | None

    @classmethod
    def loads(cls, data):
        name = uid = None
        includes = []
        references = []
        coats = {}
        for line in _text(data).splitlines():
            include = _DATA_INCLUDE.match(line)
            if include:
                includes.append(include["quoted"] or include["plain"])
                continue
            header = _DATA_HEADER.match(line)
            if header:
                name, uid = header["quoted"] or header["plain"], header["uid"]
                continue
            reference = _DATA_REF.match(line)
            if reference and reference["key"] in _REFERENCE_FIELDS:
                part, role = _REFERENCE_FIELDS[reference["key"]]
                references.append(
                    (part, role, reference["name"], reference["target"].upper())
                )
                continue
            coat = _DATA_COAT.match(line)
            if coat:
                coats[coat["key"]] = coat["path"]
        if name is None:
            raise ValueError("not a valid .mbansheepatterndata")
        return cls(
            name=name,
            uid=uid or "",
            includes=tuple(includes),
            references=tuple(references),
            body_coat=coats.get("myBodyPatternCoat"),
            head_coat=coats.get("myHeadPatternCoat"),
        )

    def member_paths(self):
        members = {}
        wanted = {
            (part, role): name.casefold()
            for part, role, name, _target in self.references
        }
        for path in self.includes:
            lower = path.casefold()
            if lower.endswith("bansheepatterndata.fruit"):
                continue
            part = "body" if "body" in lower else ("head" if "head" in lower else None)
            if part is None:
                continue
            if lower.endswith(".mcolorpattern"):
                role = "color"
            elif lower.endswith(".mpatterncontrol"):
                role = "control"
            else:
                continue
            key = (part, role)
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].casefold()
            if key not in members or stem == wanted.get(key):
                members[key] = path
        if self.body_coat:
            members[("body", "coat")] = self.body_coat
        if self.head_coat:
            members[("head", "coat")] = self.head_coat
        return members

    def reference_targets(self):
        return {
            (part, role): target
            for part, role, _name, target in self.references
        }
